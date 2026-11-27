# ======================================================================================
# Runner (appended after the generated code and tools/inputs.jl).
# One process measures one phase (environment STADE_PHASE), for every size of the kernel:
#   time        verify (CPU against device, verify sizes), then time the primal and the adjoint with two timers:
#               CUDA events (CUDA.@elapsed) and the host clock with a device synchronization. No memory reading.
#   mem_primal  only the inputs are on the device. First call, 3 calls, memory reading. No timing.
#   mem_adjoint the inputs, the gradient buffers and the stacks are on the device. First call, 3 calls, memory reading. No timing.
#   prof_primal, prof_adjoint  one size (environment STADE_SIZE, for example bench_1e6). First call, one warm-up call, then 3 calls between
#               cuProfilerStart and cuProfilerStop, so that nsys or ncu (capture range = profiler API) records only these calls. No timing.
# The reset of the inputs is outside every timer.
# ======================================================================================
using Statistics
const PHASE = get(ENV, "STADE_PHASE", "time")
PHASE in ("time", "mem_primal", "mem_adjoint", "prof_primal", "prof_adjoint") || error("STADE_PHASE must be time, mem_primal, mem_adjoint, prof_primal or prof_adjoint, got $PHASE")
const ISMEM = PHASE in ("mem_primal", "mem_adjoint")
const ISPROF = startswith(PHASE, "prof")
const HAVE_CUDA = isdefined(@__MODULE__, :CUDA)   # the CPU stand-in (tools/cpu_standin.py) defines a CUDA module too
const SUFFIX = "cuda"
_arr(x) = CuArray(x)
const ON_DEVICE = HAVE_CUDA
_sync() = (ON_DEVICE && CUDA.synchronize(); nothing)
# the profiler range: the device calls between start and stop are the ones that nsys or ncu records
_pstart() = (ON_DEVICE && Base.invokelatest(CUDA.Profile.start); nothing)
_pstop() = (ON_DEVICE && Base.invokelatest(CUDA.Profile.stop); nothing)
# CUDA.@elapsed is a macro; parse it from a string so a machine without CUDA can still load this file
const _ELAPSED = ON_DEVICE ? Core.eval(@__MODULE__, Meta.parse("f -> CUDA.@elapsed f()")) : nothing
_dtime(f) = ON_DEVICE ? Base.invokelatest(_ELAPSED, f) : _wtime(f)
# host clock around one call, closed by a device synchronization (the same quantity as the JAX timer)
function _wtime(f)
    _sync(); t0 = time_ns(); f(); _sync()
    return (time_ns() - t0) * 1e-9
end
_ckname(f) = [n for n in Base.method_argnames(first(methods(f)))[2:end] if !startswith(string(n), "#")]

function _stats(ts)
    s = sort(ts); n = length(s)
    q(p) = s[clamp(round(Int, 1 + p * (n - 1)), 1, n)]
    return Dict{String,Any}("median" => q(0.5), "q1" => q(0.25), "q3" => q(0.75), "min" => s[1], "max" => s[end], "n" => n)
end

# first call (compile on the first size), a few warm-ups, then timed calls; one event rep and one host-clock rep alternate
function _timeit(f, reset!)
    reset!(); _sync()
    t0 = _wtime(f)
    nw = t0 > 5 ? 1 : (t0 > 1 ? 2 : 4); nrep = t0 > 5 ? 5 : (t0 > 1 ? 10 : 30)
    for _ in 1:nw; reset!(); f(); end
    _sync()
    te = Float64[]; tw = Float64[]; t00 = time()
    for _ in 1:nrep
        reset!(); _sync(); push!(te, _dtime(f))
        reset!(); _sync(); push!(tw, _wtime(f))
        time() - t00 > 180 && break   # 90 s for each of the two timers
    end
    st = _stats(te); st["first_call_s"] = t0
    return st, _stats(tw)
end

# Maps the adjoint's parameters to values: primal arguments, gradient buffers (name * "b"), stacks.
function _adj_setup(fb, finit, base::Dict{Symbol,Any}, zer)
    iname = _ckname(finit)
    st = Base.invokelatest(finit, [base[s] for s in iname]...)
    st = st === nothing ? () : (st isa Tuple ? st : (st,))
    for s in st
        (s isa AbstractArray && eltype(s) <: AbstractFloat) && fill!(s, 0.0)
    end
    vals = Any[]; g = Dict{Symbol,Any}(); si = 0
    for nm in _ckname(fb)
        s = string(nm)
        if haskey(base, nm)
            push!(vals, base[nm])
        elseif occursin("_stack", s)
            si += 1; push!(vals, st[si])
        elseif endswith(s, "b") && haskey(base, Symbol(s[1:end-1]))
            b = Symbol(s[1:end-1]); buf = zer(base[b]); g[b] = buf; push!(vals, buf)
        else
            error("cannot map adjoint argument $nm")
        end
    end
    return vals, g, st
end

_zdev(x) = CUDA.zeros(eltype(x), size(x)...)
_zhost(x) = zeros(eltype(x), size(x))
_relerr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(a), abs.(b), 1.0); init = 0.0)
_nbytes(args) = sum(sizeof(v) for (_, v) in args if v isa AbstractArray; init = 0)

# device memory in use, from the driver (cuMemGetInfo): total - free, the same quantity nvidia-smi reports for the whole device
_used_dev() = ON_DEVICE ? (CUDA.total_memory() - CUDA.free_memory()) / 1e6 : nothing
_rd(a, b) = (a === nothing || b === nothing) ? nothing : a - b
_add(a, b) = (a === nothing || b === nothing) ? nothing : a + b
# bytes of the arrays that are alive in the CUDA.jl pool (exact, no pool chunk rounding)
function _live_mb()
    ON_DEVICE || return nothing
    try return CUDA.used_memory() / 1e6 catch; return nothing end
end
# CUDA.reclaim() throws under the application replay of ncu: a failed reclaim only leaves the pool as it is
_reclaim() = try CUDA.reclaim() catch; nothing end
function _settle()
    GC.gc(); ON_DEVICE && _reclaim(); GC.gc(); ON_DEVICE && _reclaim()
    nothing
end
_alloc_mb(f) = ON_DEVICE ? Base.invokelatest(Core.eval(@__MODULE__, Meta.parse("f -> CUDA.@allocated f()")), f) / 1e6 : nothing

function run_size(label, verify)
    res = Dict{String,Any}("size" => label, "phase" => PHASE)
    _settle(); mem0 = ISMEM ? _used_dev() : nothing   # only the memory phases read memory
    ISMEM && (res["mem_context_mb"] = mem0)
    args = bench_inputs(NAME, label)
    host = Dict{Symbol,Any}(args)
    rng = MersenneTwister(7)
    seed = Dict{Symbol,Vector{Float64}}(n => randn(rng, length(host[n])) for n in DEPS)
    orig = Dict{Symbol,Any}(n => copy(v) for (n, v) in args if v isa AbstractArray)
    dev = Dict{Symbol,Any}(n => (v isa AbstractArray ? _arr(v) : v) for (n, v) in args)
    res["input_mb"] = _nbytes(args) / 1e6
    if ISMEM; _settle(); res["mem_inputs_mb"] = _rd(_used_dev(), mem0); res["live_inputs_mb"] = _live_mb(); end
    fp = getfield(@__MODULE__, Symbol(NAME * "_" * SUFFIX))
    fb = getfield(@__MODULE__, Symbol(NAME * "_b_" * SUFFIX))
    fi = getfield(@__MODULE__, Symbol("initstacks_" * NAME * "_b_" * SUFFIX))
    pargs = [dev[nm] for nm in _ckname(fp)]
    reset_p!() = (for n in DEPS; copyto!(dev[n], orig[n]); end; nothing)

    # profile phases: 3 calls inside the profiler range. The reset before each call (outside the call) is inside the range too, so the kernels
    # of the reset (fill!) appear in the trace. The report sets them apart by name.
    function prof_calls(f, reset!)
        reset!(); res["first_call_s"] = @elapsed (f(); _sync())
        reset!(); f(); _sync()
        ts = Float64[]
        _pstart()
        for _ in 1:3
            reset!(); _sync(); push!(ts, _wtime(f))
        end
        _pstop()
        res["prof_calls_s"] = ts
        return res
    end
    PHASE == "prof_primal" && return prof_calls(() -> fp(pargs...), reset_p!)

    if PHASE == "mem_primal"
        # only the inputs are on the device: no gradient buffers, no stacks
        reset_p!(); res["first_primal_call_s"] = @elapsed (fp(pargs...); _sync())
        res["elements_changed"] = sum(count(Array(dev[n]) .!= orig[n]) for n in DEPS)
        _settle()
        for _ in 1:3; reset_p!(); fp(pargs...); end
        _sync(); res["mem_peak_primal_mb"] = _rd(_used_dev(), mem0)
        reset_p!(); _sync(); res["alloc_primal_call_mb"] = _alloc_mb(() -> fp(pargs...))
        res["peak_exact_primal_mb"] = _add(res["live_inputs_mb"], res["alloc_primal_call_mb"])   # arrays alive + bytes one call allocates
        return res
    end

    vals, g, st = _adj_setup(fb, fi, dev, _zdev)
    reset_a!() = begin
        for n in DEPS; copyto!(dev[n], orig[n]); end
        for (n, buf) in g
            n in DEPS ? copyto!(buf, seed[n]) : fill!(buf, 0)
        end
        nothing
    end
    res["stack_mb"] = sum((sizeof(s) for s in st if s isa AbstractArray); init = 0) / 1e6
    res["grad_mb"] = sum((sizeof(b) for (_, b) in g); init = 0) / 1e6
    res["device_mb_estimate"] = res["input_mb"] + res["grad_mb"] + res["stack_mb"]

    PHASE == "prof_adjoint" && return prof_calls(() -> fb(vals...), reset_a!)

    if PHASE == "mem_adjoint"
        _settle(); res["mem_setup_mb"] = _rd(_used_dev(), mem0)   # inputs + gradient buffers + stacks, nothing else live
        res["live_setup_mb"] = _live_mb()
        reset_a!(); res["first_adjoint_call_s"] = @elapsed (fb(vals...); _sync())
        res["grad_nonzero"] = sum(count(!=(0.0), Array(buf)) for (_, buf) in g)
        _settle()
        for _ in 1:3; reset_a!(); fb(vals...); end
        _sync(); res["mem_peak_adjoint_mb"] = _rd(_used_dev(), mem0)
        reset_a!(); _sync(); res["alloc_adjoint_call_mb"] = _alloc_mb(() -> fb(vals...))
        res["peak_exact_adjoint_mb"] = _add(res["live_setup_mb"], res["alloc_adjoint_call_mb"])
        return res
    end

    # phase time. The first call also gives the compile time (first size only).
    reset_p!(); res["first_primal_call_s"] = @elapsed (fp(pargs...); _sync())
    res["elements_changed"] = sum(count(Array(dev[n]) .!= orig[n]) for n in DEPS)
    if verify
        cp = Dict{Symbol,Any}(n => copy(v) for (n, v) in orig)
        for (n, v) in host; v isa AbstractArray || (cp[n] = v); end
        fcpu = getfield(@__MODULE__, Symbol(NAME))
        fcpu([cp[nm] for nm in _ckname(fcpu)]...)
        res["primal_relerr"] = maximum(_relerr(cp[n], Array(dev[n])) for n in DEPS)
        # adjoint: CPU against device
        hb = Dict{Symbol,Any}(n => (v isa AbstractArray ? copy(v) : v) for (n, v) in host)
        fbc = getfield(@__MODULE__, Symbol(NAME * "_b")); fic = getfield(@__MODULE__, Symbol("initstacks_" * NAME * "_b"))
        hv, hg, _ = _adj_setup(fbc, fic, hb, _zhost)
        for (n, buf) in hg; n in DEPS ? copyto!(buf, seed[n]) : fill!(buf, 0); end
        fbc(hv...)
        reset_a!(); res["first_adjoint_call_s"] = @elapsed (fb(vals...); _sync())
        res["grad_relerr"] = maximum(_relerr(hg[n], Array(g[n])) for n in keys(g))
    else
        reset_a!(); res["first_adjoint_call_s"] = @elapsed (fb(vals...); _sync())
    end
    res["grad_nonzero"] = sum(count(!=(0.0), Array(buf)) for (_, buf) in g)
    if PROFILE
        # per-kernel device times of one primal call and one adjoint call (CUDA.jl integrated profiler)
        prof = Core.eval(@__MODULE__, Meta.parse("f -> CUDA.@profile f()"))
        for (tag, f, rs) in (("primal", () -> fp(pargs...), reset_p!), ("adjoint", () -> fb(vals...), reset_a!))
            rs(); _sync()
            try
                r = Base.invokelatest(prof, f)
                io = IOBuffer(); show(io, r)
                res["profile_" * tag] = String(take!(io))
            catch err
                res["profile_" * tag] = "profile failed: " * first(sprint(showerror, err), 300)
            end
        end
        return res
    end
    res["primal"], res["primal_wall"] = _timeit(() -> fp(pargs...), reset_p!)
    res["adjoint"], res["adjoint_wall"] = _timeit(() -> fb(vals...), reset_a!)
    res["ratio_adjoint_primal"] = res["adjoint"]["median"] / res["primal"]["median"]
    res["ratio_adjoint_primal_wall"] = res["adjoint_wall"]["median"] / res["primal_wall"]["median"]
    return res
end

function _versions()
    d = Dict{String,Any}("julia" => string(VERSION), "stade_commit" => get(ENV, "STADE_COMMIT", ""), "kit_hash" => get(ENV, "KIT_HASH", ""))
    if ON_DEVICE
        try d["cuda_jl"] = string(pkgversion(CUDA)) catch; end
        try d["driver"] = string(CUDA.driver_version()) catch; end
        try d["runtime"] = string(CUDA.runtime_version()) catch; end
    end
    return d
end

# ---- JSON writer (no package needed) ----
function _jstr(s)
    io = IOBuffer(); write(io, '"')
    for c in String(s)
        c == '"' ? write(io, "\\\"") : c == '\\' ? write(io, "\\\\") : c == '\n' ? write(io, "\\n") :
        c == '\t' ? write(io, "\\t") : (c < ' ' ? write(io, ' ') : write(io, c))
    end
    write(io, '"'); return String(take!(io))
end
_json(x::AbstractString) = _jstr(x)
_json(x::Symbol) = _jstr(string(x))
_json(x::Bool) = x ? "true" : "false"
_json(x::Nothing) = "null"
_json(x::Integer) = string(x)
_json(x::AbstractFloat) = isfinite(x) ? string(Float64(round(x; sigdigits = 6))) : "null"
_json(x::AbstractVector) = "[" * join((_json(v) for v in x), ",") * "]"
_json(x::AbstractDict) = "{" * join((_jstr(string(k)) * ":" * _json(v) for (k, v) in x), ",") * "}"

results = Any[]
# memory phases: one throwaway pass on the first size, so the one-time memory of code modules and libraries is not in the readings
if ISMEM
    wl = filter(x -> !x[2], SIZES)
    isempty(wl) || (try; run_size(first(wl)[1], false); catch; end; _settle())
end
# the memory phases skip the verify sizes (the time phase checks them)
# a profile phase runs the one size named in STADE_SIZE
_psz = get(ENV, "STADE_SIZE", "")
ISPROF && ((!isempty(_psz) && any(x -> x[1] == _psz, SIZES)) || error("STADE_SIZE=\"$_psz\" is not a size of $NAME"))
for (label, verify) in (PHASE == "time" ? SIZES : ISPROF ? filter(x -> x[1] == _psz, SIZES) : filter(x -> !x[2], SIZES))
    r = try
        run_size(label, verify)
    catch err
        Dict{String,Any}("size" => label, "error" => first(sprint(showerror, err), 600))
    end
    push!(results, r)
    GC.gc(); ON_DEVICE && _reclaim()
end
out = Dict{String,Any}("kernel" => NAME, "backend" => string(BACKEND), "fuse" => FUSE, "host" => gethostname(), "phase" => PHASE,
                       "config" => get(ENV, "BENCH_CONFIG", ""), "versions" => _versions(),
                       "device" => ON_DEVICE ? CUDA.name(CUDA.device()) : "cpu", "results" => results)
println(_json(out))
