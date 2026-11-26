
# ======================================================================================
# Runner (appended after the generated code and tools/inputs.jl).
# Per size: build inputs, check GPU against CPU (verify sizes only), time primal and adjoint.
# Timing: device events (CUDA.@elapsed) around one call; the reset of inputs is outside the event.
# ======================================================================================
using Statistics
const HAVE_CUDA = isdefined(@__MODULE__, :CUDA) && !isdefined(CUDA, :FAKE_CUDA)   # the local test shim defines a fake CUDA
const SUFFIX = BACKEND === :cuda ? "cuda" : "jacc"
_arr(x) = BACKEND === :cuda ? CuArray(x) : JACC.array(x)
const ON_DEVICE = HAVE_CUDA && (BACKEND === :cuda || (typeof(JACC.array(zeros(1))) <: CUDA.CuArray))
_sync() = (ON_DEVICE && CUDA.synchronize(); nothing)
# CUDA.@elapsed is a macro; parse it from a string so a machine without CUDA can still load this file
const _ELAPSED = ON_DEVICE ? Core.eval(@__MODULE__, Meta.parse("f -> CUDA.@elapsed f()")) : nothing
function _dtime(f)
    ON_DEVICE && return Base.invokelatest(_ELAPSED, f)
    t0 = time_ns(); f(); return (time_ns() - t0) * 1e-9
end
_ckname(f) = [n for n in Base.method_argnames(first(methods(f)))[2:end] if !startswith(string(n), "#")]
_isfloatarr(v) = v isa AbstractArray && eltype(v) <: AbstractFloat

function _stats(ts)
    s = sort(ts); n = length(s)
    q(p) = s[clamp(round(Int, 1 + p * (n - 1)), 1, n)]
    return Dict{String,Any}("median" => q(0.5), "q1" => q(0.25), "q3" => q(0.75), "min" => s[1], "max" => s[end], "n" => n)
end

# first call (compile on the first size), a few warm-ups, then timed calls
function _timeit(f, reset!)
    reset!(); _sync()
    t0 = _dtime(f)
    nw = t0 > 5 ? 1 : (t0 > 1 ? 2 : 4); nrep = t0 > 5 ? 5 : (t0 > 1 ? 10 : 30)
    for _ in 1:nw; reset!(); f(); end
    _sync()
    ts = Float64[]; tw = time()
    for _ in 1:nrep
        reset!(); _sync()
        push!(ts, _dtime(f))
        time() - tw > 90 && break
    end
    st = _stats(ts); st["first_call_s"] = t0
    return st
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

_zdev(x) = (z = similar(x); fill!(z, 0); z)
_zhost(x) = zeros(eltype(x), size(x))
_relerr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(a), abs.(b), 1.0); init = 0.0)
_nbytes(args) = sum(sizeof(v) for (_, v) in args if v isa AbstractArray; init = 0)

function _mem_gb()
    ON_DEVICE || return nothing
    try return (CUDA.total_memory() - CUDA.free_memory()) / 1e9 catch; return nothing end
end

# device memory in use, from the driver (cuMemGetInfo): total - free, the same quantity nvidia-smi reports for the whole device
_used_dev() = ON_DEVICE ? (CUDA.total_memory() - CUDA.free_memory()) / 1e6 : nothing
function _settle()
    GC.gc(); ON_DEVICE && CUDA.reclaim(); GC.gc(); ON_DEVICE && CUDA.reclaim()
    nothing
end

function run_size(label, verify)
    res = Dict{String,Any}("size" => label)
    _settle(); mem0 = _used_dev(); res["mem_context_mb"] = mem0
    args = bench_inputs(NAME, label)
    host = Dict{Symbol,Any}(args)
    rng = MersenneTwister(7)
    seed = Dict{Symbol,Vector{Float64}}(n => randn(rng, length(host[n])) for n in DEPS)
    orig = Dict{Symbol,Any}(n => copy(v) for (n, v) in args if v isa AbstractArray)
    dev = Dict{Symbol,Any}(n => (v isa AbstractArray ? _arr(v) : v) for (n, v) in args)
    res["input_mb"] = _nbytes(args) / 1e6
    _settle(); res["mem_inputs_mb"] = _used_dev() - mem0
    if HAVE_CUDA
        try; GC.gc(); CUDA.reclaim(); res["live_inputs_mb"] = CUDA.used_memory() / 1e6; catch err; res["live_inputs_mb"] = -1.0; end
    end
    fp = getfield(@__MODULE__, Symbol(NAME * "_" * SUFFIX))
    fb = getfield(@__MODULE__, Symbol(NAME * "_b_" * SUFFIX))
    fi = getfield(@__MODULE__, Symbol("initstacks_" * NAME * "_b_" * SUFFIX))
    pargs = [dev[nm] for nm in _ckname(fp)]
    reset_p!() = (for n in DEPS; copyto!(dev[n], orig[n]); end; nothing)
    vals, g, st = _adj_setup(fb, fi, dev, _zdev)
    reset_a!() = begin
        for n in DEPS; copyto!(dev[n], orig[n]); end
        for (n, buf) in g
            n in DEPS ? copyto!(buf, seed[n]) : fill!(buf, 0)
        end
        nothing
    end
    _settle(); res["mem_setup_mb"] = _used_dev() - mem0   # inputs + gradient buffers + stacks, nothing else live
    res["stack_mb"] = sum((sizeof(s) for s in st if s isa AbstractArray); init = 0) / 1e6
    res["grad_mb"] = sum((sizeof(b) for (_, b) in g); init = 0) / 1e6
    res["device_mb_estimate"] = res["input_mb"] + res["grad_mb"] + res["stack_mb"]

    # gate 5: the run changes elements. The first call also gives the compile time (first size only).
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
    if HAVE_CUDA
        # device memory: the arrays live on the device, and what one call of the primal and of the adjoint allocates on top of them
        try
            GC.gc(); CUDA.reclaim()
            res["live_adjoint_mb"] = CUDA.used_memory() / 1e6
            alloc = Core.eval(@__MODULE__, Meta.parse("f -> CUDA.@allocated f()"))
            reset_p!(); _sync(); res["alloc_primal_call_mb"] = Base.invokelatest(alloc, () -> fp(pargs...)) / 1e6
            reset_a!(); _sync(); res["alloc_adjoint_call_mb"] = Base.invokelatest(alloc, () -> fb(vals...)) / 1e6
            GC.gc()
        catch err
            res["mem_error"] = first(sprint(showerror, err), 200)
        end
    end

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
    res["primal"] = _timeit(() -> fp(pargs...), reset_p!)
    res["adjoint"] = _timeit(() -> fb(vals...), reset_a!)
    res["ratio_adjoint_primal"] = res["adjoint"]["median"] / res["primal"]["median"]
    # memory pass: separate calls after the timing, so the timings stay clean. The pool keeps what it grew to, so the reading after the calls is the high-water mark.
    try
        _settle()
        for _ in 1:3; reset_p!(); fp(pargs...); end
        _sync(); res["mem_peak_primal_mb"] = _used_dev() - mem0
        _settle()
        for _ in 1:3; reset_a!(); fb(vals...); end
        _sync(); res["mem_peak_adjoint_mb"] = _used_dev() - mem0
    catch err
        res["mem_error"] = first(sprint(showerror, err), 200)
    end
    return res
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
for (label, verify) in SIZES
    r = try
        run_size(label, verify)
    catch err
        Dict{String,Any}("size" => label, "error" => first(sprint(showerror, err), 600))
    end
    push!(results, r)
    GC.gc(); ON_DEVICE && CUDA.reclaim()
end
out = Dict{String,Any}("kernel" => NAME, "backend" => string(BACKEND), "fuse" => FUSE, "host" => gethostname(),
                       "device" => ON_DEVICE ? CUDA.name(CUDA.device()) : "cpu", "results" => results)
println(_json(out))
