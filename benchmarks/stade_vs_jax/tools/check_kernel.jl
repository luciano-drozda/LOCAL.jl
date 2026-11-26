# Checks one benchmark kernel with STADE's own generators and oracles. No GPU needed.
#   julia check_kernel.jl NAME [SIZE]        (SIZE is a key of the size table in inputs.jl)
# For each of four flag combinations it generates the adjoint, then checks
#   1. adjoint against a central finite difference along a random direction,
#   2. adjoint against the tangent code (dot-product identity, exact to rounding),
# and for keep_push_pop = false it also generates the CUDA and JACC code and reports the loops
# that stay on the host.
# STADE_SRC must point at STADE.jl/src/STADE.jl.  CK_WORK, if set, is the directory for generated code.
haskey(ENV, "STADE_SRC") || error("set STADE_SRC to the path of STADE.jl/src/STADE.jl")
include(ENV["STADE_SRC"])
using .STADE, Random, LinearAlgebra, Printf
include(joinpath(@__DIR__, "inputs.jl"))

ck_dup(v) = v isa AbstractArray ? copy(v) : v
ck_isfloatarr(v) = v isa AbstractArray && eltype(v) <: AbstractFloat

# Load kernel `name` plus its generated derivative code into a fresh module.
function ck_load(path, name, gen; kpp, fuse, tag)
    dst = joinpath(get(ENV, "CK_WORK", mktempdir()), "$(name)_$(gen)_$(tag).jl")
    gen == :b ? STADE.stade_adjoint_file(path, dst; keep_push_pop = kpp, fuse_ii_loops = fuse) :
                STADE.stade_tangent_file(path, dst; keep_push_pop = kpp, fuse_ii_loops = fuse)
    m = Module(:Chk)
    Base.include(m, dst)
    return m, dst
end

function ck_names(f)
    return [n for n in Base.method_argnames(first(methods(f)))[2:end] if !startswith(string(n), "#")]
end

# Arrays the kernel writes (the ones that receive a cotangent) and all float arrays (the ones
# differentiated). STADE's own sig.dependents lists every float argument, so the written set is
# read from the assignments in the kernel body.
function ck_roles(path)
    kernel = STADE.parse_kernel(STADE.io_read_corpus_entry(path))
    written = Set{Symbol}()
    function walk(stmts)
        for st in stmts
            if st.kind == :assign
                st.lhs isa Symbol || push!(written, st.lhs.args[1])
            elseif st.kind == :for || st.kind == :while
                walk(st.body)
            elseif st.kind == :if
                walk(st.then); walk(st.els)
            end
        end
    end
    walk(kernel.body)
    floats = Set(n for n in kernel.sig.args if kernel.sig.kinds[n] == :array_float)
    return intersect(written, floats), floats
end

function ck_run_primal(m, name, args)
    vals = Any[ck_dup(v) for (_, v) in args]
    Base.invokelatest(getfield(m, Symbol(name)), vals...)
    return Dict(zip(first.(args), vals))
end

# Adjoint run. `seed`: cotangent per dependent array. Returns gradient per float array argument.
function ck_run_adjoint(m, name, args, seed, deps)
    fb = getfield(m, Symbol(name * "_b")); init = getfield(m, Symbol("initstacks_" * name * "_b"))
    primal = Dict(args)
    iname = ck_names(init)
    stacks = Base.invokelatest(init, [primal[s] for s in iname]...)
    stacks = stacks === nothing ? () : (stacks isa Tuple ? stacks : (stacks,))
    si = 0; vals = Any[]; adj = Dict{Symbol,Any}()
    for nm in ck_names(fb)
        s = string(nm)
        if haskey(primal, nm); push!(vals, ck_dup(primal[nm]))
        elseif occursin("_stack", s); si += 1; push!(vals, stacks[si])
        elseif endswith(s, "b") && haskey(primal, Symbol(s[1:end-1]))
            base = Symbol(s[1:end-1]); pv = primal[base]
            buf = base in deps ? copy(seed[base]) : zeros(size(pv)); adj[base] = buf; push!(vals, buf)
        else error("cannot map adjoint argument $nm") end
    end
    Base.invokelatest(fb, vals...)
    return adj
end

# Tangent run along direction `dir` (per independent array). Returns tangent per dependent array.
function ck_run_tangent(m, name, args, dir, deps)
    fd = getfield(m, Symbol(name * "_d")); primal = Dict(args)
    vals = Any[]; tan = Dict{Symbol,Any}()
    for nm in ck_names(fd)
        s = string(nm)
        if haskey(primal, nm); push!(vals, ck_dup(primal[nm]))
        elseif endswith(s, "d") && haskey(primal, Symbol(s[1:end-1]))
            base = Symbol(s[1:end-1]); pv = primal[base]
            buf = haskey(dir, base) ? copy(dir[base]) : zeros(size(pv)); tan[base] = buf; push!(vals, buf)
        else error("cannot map tangent argument $nm") end
    end
    Base.invokelatest(fd, vals...)
    return tan
end

function check_kernel(path, name, args; seed = 1, verbose = true)
    deps, indeps = ck_roles(path)
    rng = MersenneTwister(seed)
    floatarrs = [n for (n, v) in args if ck_isfloatarr(v)]
    seedv = Dict(n => randn(rng, length(Dict(args)[n])) for n in floatarrs if n in deps)
    dir = Dict(n => randn(rng, length(Dict(args)[n])) for n in floatarrs if n in indeps)
    w = Dict{Symbol,Any}(); tag = 0
    results = NamedTuple[]
    mp = Module(:Prim); Base.include(mp, path)
    # finite difference along dir: f(x) = sum over dependents of seed . y
    function f(eps)
        a = Pair{Symbol,Any}[n => (haskey(dir, n) ? v .+ eps .* dir[n] : v) for (n, v) in args]
        out = ck_run_primal(mp, name, a)
        return sum(dot(seedv[n], out[n]) for n in keys(seedv))
    end
    h = 1e-6
    fdv = (f(h) - f(-h)) / (2h)
    for kpp in (true, false), fuse in (false, true)
        tag += 1
        t0 = time()
        mb, fb = ck_load(path, name, :b; kpp, fuse, tag)
        adj = ck_run_adjoint(mb, name, args, seedv, deps)
        ad = sum(dot(adj[n], dir[n]) for n in keys(dir))
        mt, _ = ck_load(path, name, :d; kpp, fuse, tag)
        tan = ck_run_tangent(mt, name, args, dir, deps)
        td = sum(dot(seedv[n], tan[n]) for n in keys(seedv))
        e_fd = abs(ad - fdv) / max(abs(fdv), 1e-300)
        e_dot = abs(ad - td) / max(abs(td), 1e-300)
        gpu = missing
        kpp || (gpu = ck_offload(fb))
        push!(results, (kpp = kpp, fuse = fuse, e_fd = e_fd, e_dot = e_dot, gpu = gpu, secs = time() - t0, file = fb))
        verbose && @printf("  %-5s fuse=%-5s  adjoint-vs-FD %.2e   adjoint-vs-tangent %.2e   (%.1fs)\n",
                           kpp ? "kpp" : "nopp", string(fuse), e_fd, e_dot, time() - t0)
    end
    return results
end

# Loops in the generated adjoint, loops left on the host after CUDA conversion, device kernels,
# and whether the JACC path also converts. Same counting as test/validate_offload.jl.
function ck_offload(file)
    count_for(e) = e isa Expr ? (e.head == :for) + sum(Int[count_for(a) for a in e.args]; init = 0) : 0
    total = 0; host = 0; kernels = 0; jacc = true
    try
        for e in STADE.io_read_kernel_bundle(file)
            total += count_for(e)
            plan = STADE.stade_gpu(e, STADE.cgen_backend_cuda())
            host += count_for(plan.host); kernels += length(plan.kernels)
            try STADE.stade_jacc(e) catch; jacc = false end
        end
    catch err
        return (refused = first(split(sprint(showerror, err), "\n")),)
    end
    return (loops = total, host = host, kernels = kernels, jacc = jacc)
end

if abspath(PROGRAM_FILE) == (@__FILE__) && basename(PROGRAM_FILE) == "check_kernel.jl"
    name = ARGS[1]; szname = length(ARGS) >= 2 ? ARGS[2] : "tiny"
    path = bench_kernel_path(name)
    args = bench_inputs(name, szname)
    println("kernel ", name, "  size ", szname)
    res = check_kernel(path, name, args)
    bad = [r for r in res if r.e_fd > 1e-5 || r.e_dot > 1e-11]
    for r in res
        r.kpp || println("  GPU path, fuse=$(r.fuse): ", r.gpu)
    end
    println(isempty(bad) ? "PASS $name" : "FAIL $name")
end
