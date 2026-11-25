# Memory and time of the checkpointed adjoint against the plain fixed-point one on a device (or a stand-in): phc_heat_steps, n points, nsteps steps.
# Reports the bytes of the stacks initstacks allocates, the device memory in use after the run, the time of the adjoint, and the worst relative difference of the shadows.
#   julia measure_checkpoint_device.jl [n nsteps ck...]   (stand-in); a device driver binds scc_env and spj_env and calls ckm_measure with sg_cuda_env() or sg_jacc_env().
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))
isdefined(@__MODULE__, :sg_rename) || include(joinpath(@__DIR__, "validate_adjoint_split_gpu.jl"))
isdefined(@__MODULE__, :scc_env) || include(joinpath(@__DIR__, "validate_adjoint_split_cuda_cpu.jl"))
using Random, Logging

# one run: (stack bytes, time in s of the adjoint call, shadows of the arrays, fp_info)
function ckm_one(env, primal::Expr, n::Int, nsteps::Int, maxit::Int, ck, devmem; checkpoint::Bool)
    r = with_logger(NullLogger()) do; STADE.stade_adjoint(primal; keep_push_pop = false, checkpoint = checkpoint) end
    ksig = STADE.parse_kernel(checkpoint ? r.primal : primal).sig
    snames = Symbol[a for a in r.initstacks.args[1].args[2:end]]
    ints = Dict{Symbol,Int}(:n => n, :nsteps => nsteps, :maxit => maxit, :ck => something(ck, 1))
    Random.seed!(7)
    vals = Dict{Symbol,Any}(:u => [sin(0.37 * i) + 0.5 * cos(0.011 * i) for i in 1:n], :w => zeros(n), :wn => zeros(n), :lam => 0.4, :tol => 1e-20)
    seedv = Dict{Symbol,Any}(:u => STADE.val_randn(n), :w => zeros(n), :wn => zeros(n), :lam => 0.0, :tol => 0.0)
    plan_init = env.plan(r.initstacks, nothing)
    plan_adj = env.plan(r.adjoint, env.initmap(plan_init, r.initstacks))
    dev_init = sg_eval_plan!(env.mod, plan_init); dev_adj = sg_eval_plan!(env.mod, plan_adj)
    a, t = sg_build_call(ksig, ints, vals, seedv, env.wrap)
    st = sg_init_stacks(dev_init, sg_stack_extra(snames, ints, vals, env.wrap))
    bytes = sum(s -> s isa AbstractArray ? sizeof(s) : 0, st; init = 0)
    Base.invokelatest(dev_adj, a..., st...)    # warm-up compiles the kernels; the arrays are reseeded below
    a, t = sg_build_call(ksig, ints, vals, seedv, env.wrap)
    st = sg_init_stacks(dev_init, sg_stack_extra(snames, ints, vals, env.wrap))
    devmem(:sync)
    t0 = time(); Base.invokelatest(dev_adj, a..., st...); devmem(:sync); dt = time() - t0
    return (bytes = bytes, seconds = dt, shadow = env.tohost(t[:u].sh), mem = devmem(:used))
end

function ckm_measure(env, n::Int, nsteps::Int, cks; devmem = m -> 0, maxit::Int = 400, plain::Bool = true)
    src = read(joinpath(@__DIR__, "counted-corpus", "phc_heat_steps.jl"), String)
    primal = Meta.parse(src)
    rows = Any[]
    ref = nothing
    if plain
        base = ckm_one(env, primal, n, nsteps, maxit, nothing, devmem; checkpoint = false)
        push!(rows, (ck = 0, bytes = base.bytes, seconds = base.seconds, mem = base.mem, err = 0.0)); ref = base.shadow
    end
    for ck in cks
        c = ckm_one(env, primal, n, nsteps, maxit, ck, devmem; checkpoint = true)
        ref === nothing && (ref = c.shadow)
        push!(rows, (ck = ck, bytes = c.bytes, seconds = c.seconds, mem = c.mem, err = sg_relerr(ref, c.shadow)))
    end
    return rows
end

# one HVP run: the stacks of initstacks hold the primal tape and the tangent checkpoints; devalloc(f) runs f and returns the device bytes it allocated (the HVP allocates its shadow tape inside)
function ckm_one_hvp(env, primal::Expr, n::Int, nsteps::Int, maxit::Int, ck, devmem, devalloc; checkpoint::Bool)
    r = with_logger(NullLogger()) do; STADE.stade_hvp(primal; keep_push_pop = false, checkpoint = checkpoint) end
    ksig = STADE.parse_kernel(checkpoint ? r.primal : primal).sig
    snames = Symbol[a for a in r.initstacks.args[1].args[2:end]]
    ints = Dict{Symbol,Int}(:n => n, :nsteps => nsteps, :maxit => maxit, :ck => something(ck, 1))
    u0 = [sin(0.37 * i) + 0.5 * cos(0.011 * i) for i in 1:n]
    vals = Dict{Symbol,Any}(:u => u0, :w => zeros(n), :wn => zeros(n), :lam => 0.4, :tol => 1e-20)
    seedv = Dict{Symbol,Any}(:u => STADE.val_randn(Random.Xoshiro(7), n), :w => zeros(n), :wn => zeros(n), :lam => 0.0, :tol => 0.0)
    D = Dict{Symbol,Any}(:u => STADE.val_randn(Random.Xoshiro(8), n), :w => zeros(n), :wn => zeros(n), :lam => 0.0, :tol => 0.0)
    dseed = Dict{Symbol,Any}(:u => zeros(n), :w => zeros(n), :wn => zeros(n), :lam => 0.0, :tol => 0.0)
    plan_init = env.plan(r.initstacks, nothing)
    plan_hvp = env.plan(r.hvp, env.initmap(plan_init, r.initstacks))
    dev_init = sg_eval_plan!(env.mod, plan_init); dev_hvp = sg_eval_plan!(env.mod, plan_hvp)
    extras() = Any[haskey(ints, nm) ? ints[nm] : (vals[nm] isa AbstractArray ? env.wrap(deepcopy(vals[nm])) : vals[nm]) for nm in snames]
    mk() = begin
        st = STADE.val_init_stacks(dev_init, extras())
        a = STADE.val_hvp_call_args(ksig, ints, vals, seedv, D, dseed, st)
        for i in 1:(length(a) - length(st)); a[i] isa AbstractArray && (a[i] = env.wrap(a[i])); end
        return (a, st)
    end
    (a, st) = mk()
    bytes = sum(x -> x isa AbstractArray ? sizeof(x) : 0, st; init = 0)
    Base.invokelatest(dev_hvp, a...)    # warm-up compiles the kernels
    (a, st) = mk()
    devmem(:sync)
    t0 = time(); alloc = devalloc(() -> Base.invokelatest(dev_hvp, a...)); devmem(:sync); dt = time() - t0
    outs = [env.tohost(x) for x in a[1:(length(a) - length(st))] if x isa AbstractArray && eltype(x) === Float64]
    return (bytes = bytes, seconds = dt, outs = outs, alloc = alloc, mem = devmem(:used))
end

function ckm_hvp_err(ref, got)
    err = 0.0; signal = 0.0
    for (x, y) in zip(ref, got)
        signal = max(signal, maximum(abs, x; init = 0.0))
        err = max(err, sg_relerr(x, y))
    end
    signal > 0 || error("every array of the HVP is zero")
    return err
end

function ckm_measure_hvp(env, n::Int, nsteps::Int, cks; devmem = m -> 0, devalloc = f -> (f(); 0), maxit::Int = 400, plain::Bool = true)
    primal = Meta.parse(read(joinpath(@__DIR__, "counted-corpus", "phc_heat_steps.jl"), String))
    rows = Any[]; ref = nothing
    if plain
        b = ckm_one_hvp(env, primal, n, nsteps, maxit, nothing, devmem, devalloc; checkpoint = false)
        push!(rows, (ck = 0, bytes = b.bytes, alloc = b.alloc, seconds = b.seconds, mem = b.mem, err = 0.0)); ref = b.outs
    end
    for ck in cks
        c = ckm_one_hvp(env, primal, n, nsteps, maxit, ck, devmem, devalloc; checkpoint = true)
        ref === nothing && (ref = c.outs)
        push!(rows, (ck = ck, bytes = c.bytes, alloc = c.alloc, seconds = c.seconds, mem = c.mem, err = ckm_hvp_err(ref, c.outs)))
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    a = [parse(Int, x) for x in ARGS]
    n, nsteps = isempty(a) ? (200, 12) : (a[1], a[2])
    for r in ckm_measure(scc_env(), n, nsteps, length(a) > 2 ? a[3:end] : [1, 3, 4]); println(r); end
    for r in ckm_measure_hvp(scc_env(), n, nsteps, length(a) > 2 ? a[3:end] : [1, 3, 4]); println("hvp ", r); end
end
