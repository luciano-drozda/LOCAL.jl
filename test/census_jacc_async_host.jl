# Census of the host code of stade_jacc(...; jacc_async = true) over the corpus (v0.5.26). Plan only, nothing runs.
# For each kernel, adjoint and HVP, glue off and on, production flags otherwise, it walks the generated host function and records
#   - every launch (`JACC.@parallel_for`, `JACC.launch_spec`) and whether it carries `sync = false`,
#   - every other call and macro that appears on the host, and every index expression `x[...]` outside a launch (a host read or write of a device array),
#   - which of those follow an async launch without a launch-free path (see `after_async`).
# Output: one line per kernel `CENSUS {json}`, then a merged table `FORMS {name: count}`. A form not in the audited list of
# `async_audited_forms` is printed as UNAUDITED: new host forms must be argued before jacc_async can be a default.
#   julia census_jacc_async_host.jl [kernel names...]     (CENSUS_SHARD=k/N runs every N-th kernel)

isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

# every corpus kernel by name (the same walk as validate_serial_loops.jl)
function corpus_paths()
    paths = Dict{String,String}()
    for d in sort(filter(isdir, readdir(@__DIR__; join = true))), f in sort(readdir(d; join = true))
        endswith(f, ".jl") && occursin("-corpus", d) || continue
        n = splitext(basename(f))[1]
        occursin(r"_(b|d|hv|bd|cuda|jacc)$", n) && continue
        haskey(paths, n) || (paths[n] = f)
    end
    return paths
end

# The host forms that touch device data and were audited for jacc_async (HANDOFF item 4g). On the V100 with JACC 1.3.1, each was tried right after a long
# `sync = false` launch, 12 times, and always saw the kernel's result (results/v100-async/APROBE1.json, APROBE2.json). This is a measurement, not a proof about the
# CUDA extension of JACC: a new JACC version or backend needs the probe again.
function async_audited_forms()
    return Dict{String,String}(
        "launch:@parallel_for sync=false" => "same stream, kernels run in order",
        "launch:launch_spec sync=false" => "same stream, kernels run in order",
        "launch:@parallel_reduce (waits)" => "the reduction launches on the same stream and returns a device array; probed after an async kernel",
        "launch:@parallel_for (waits)" => "a waiting launch is always safe",
        "launch:launch_spec (waits)" => "a waiting launch is always safe",
        "call:JACC.parallel_for" => "the launch; its spec is counted as launch_spec",
        "macro:@allowscalar" => "a scalar read or write copies on the same stream; probed (first and last element, an if, a write after a launch)",
        "call:JACC.to_host" => "copies on the same stream; probed",
        "call:Array" => "copies on the same stream; probed",
        "call:Vector{Float64}" => "a copy of a host array, or of to_host (probed)",
        "call:Vector{Int64}" => "a copy of a host array, or of to_host",
        "call:JACC.array" => "a host to device copy on the same stream; probed after a launch",
        "call:JACC.zeros" => "allocates and fills on the same stream; probed after a launch",
        "call:CUDA.CuArray{Float64}" => "a host to device copy on the same stream (the form of JACC.array)",
        "call:CUDA.unsafe_free!" => "the free is ordered on the stream; probed (free, reallocate, fill, then read the kernel result)",
        "call:JACC.default_backend" => "a query, no device access",
        "call:Base.cld" => "integer arithmetic",
    )
end

# a call that can touch device data without being on the audited list
function async_is_device_call(fs::String)
    fs in ("copyto!", "fill!", "copy", "unsafe_copyto!", "similar", "deepcopy", "sum", "maximum", "minimum", "reduce", "mapreduce", "count", "any", "all", "collect",
           "convert", "vec", "getindex", "setindex!", "synchronize", "unsafe_free!", "view", "reshape", "zeros", "ones", "fill", "Array", "Vector{Float64}", "Vector{Int64}") && return true
    return occursin('.', fs)
end

# the forms of a census that are neither audited nor plain host code
function async_unaudited(forms::Dict{String,Int}; aud::Dict{String,String} = async_audited_forms())
    out = String[]
    for k in sort(collect(keys(forms)))
        haskey(aud, k) && continue
        if startswith(k, "call:")
            async_is_device_call(k[6:end]) && push!(out, k)
        elseif startswith(k, "macro:") || startswith(k, "launch:")
            push!(out, k)
        end
    end
    return out
end

function census_walk!(e, forms::Dict{String,Int}; in_launch::Bool = false, async_launches::Ref{Int} = Ref(0), sync_launches::Ref{Int} = Ref(0))
    bump(k) = forms[k] = get(forms, k, 0) + 1
    e isa Expr || return
    h = e.head
    if h === :macrocall
        nm = e.args[1]
        s = nm isa Symbol ? string(nm) : nm isa Expr && nm.args[end] isa QuoteNode ? string(nm.args[end].value) : string(nm)
        s = replace(s, r"^JACC\." => "")
        if s == "@parallel_for" || s == "@parallel_reduce" || s == "@parallel_for"
            asy = any(a -> a isa Expr && a.head === :(=) && a.args[1] === :sync && a.args[2] === false, e.args)
            bump("launch:" * s * (asy ? " sync=false" : " (waits)"))
            asy ? (async_launches[] += 1) : (sync_launches[] += 1)
            for a in e.args[2:end]; census_walk!(a, forms; in_launch = true, async_launches, sync_launches); end
            return
        end
        bump("macro:" * s)
    elseif h === :call
        f = e.args[1]
        fs = f isa Symbol ? string(f) : f isa Expr && f.head === :. ? join([a isa QuoteNode ? string(a.value) : string(a) for a in f.args], ".") : string(f)
        if fs == "JACC.launch_spec"
            asy = any(a -> a isa Expr && a.head === :parameters && any(k -> k isa Expr && k.head === :kw && k.args[1] === :sync && k.args[2] === false, a.args), e.args)
            bump("launch:launch_spec" * (asy ? " sync=false" : " (waits)"))
            asy ? (async_launches[] += 1) : (sync_launches[] += 1)
        elseif !in_launch
            bump("call:" * fs)
        end
    elseif h === :ref && !in_launch
        bump("index:" * (e.args[1] isa Symbol ? "x[..]" : "expr[..]"))
    end
    for a in e.args; census_walk!(a, forms; in_launch, async_launches, sync_launches); end
end

function census_kernel(name, primal)
    out = Dict{String,Int}(); nasync = 0; nsync = 0
    for mode in (:adjoint, :hvp), glue in (false, true)
        gen = mode == :hvp ? STADE.stade_hvp(primal; keep_push_pop = false, fuse_ii_loops = true) : STADE.stade_adjoint(primal; keep_push_pop = false, fuse_ii_loops = true)
        ex = mode == :hvp ? gen.hvp : gen.adjoint
        pi = STADE.stade_jacc(gen.initstacks; jacc_async = true, glue_kernels = glue)
        pg = STADE.stade_jacc(ex; jacc_async = true, glue_kernels = glue, init_hosts = Dict{Symbol,Expr}(pi.host.args[1].args[1] => pi.host))
        for pl in (pi, pg)
            a = Ref(0); s = Ref(0)
            census_walk!(pl.host, out; async_launches = a, sync_launches = s)
            nasync += a[]; nsync += s[]
        end
    end
    return out, nasync, nsync
end

function main(names)
    paths = corpus_paths()
    sel = isempty(names) ? sort(collect(keys(paths))) : names
    shard = get(ENV, "CENSUS_SHARD", "")
    if isempty(names) && !isempty(shard)
        (k, nn) = parse.(Int, split(shard, "/"))
        sel = sel[[i for i in eachindex(sel) if mod(i - 1, nn) == k]]
    end
    merged = Dict{String,Int}(); nerr = 0; nk = 0; tot_async = 0; tot_sync = 0
    for n in sel
        try
            (f, na, ns) = census_kernel(n, STADE.io_read_corpus_entry(paths[n]))
            for (k, v) in f; merged[k] = get(merged, k, 0) + v; end
            tot_async += na; tot_sync += ns; nk += 1
        catch e
            nerr += 1
            println("ERROR $n: ", first(sprint(showerror, e), 200))
        end
    end
    println("FORMS ", join(["$k=$v" for (k, v) in sort(collect(merged))], " | "))
    ua = async_unaudited(merged)
    isempty(ua) || println("UNAUDITED ", join(ua, " | "))
    println("TOTALS kernels=$nk errors=$nerr async_launches=$tot_async waiting_launches=$tot_sync")
    return isempty(ua)
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main(String.(ARGS)) ? println("DONE") : (println("FAILED"); exit(1))
end
