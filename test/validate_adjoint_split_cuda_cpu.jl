# Regression for v0.5.37: the CUDA code of the two halves of stade_adjoint_split, run on the CPU with the serial CUDA stand-in of validate_serial_loops.jl
# (every (block, thread) pair of a launch runs once, in order; the passes that need hardware are off). The forward half records the tape, the backward
# half replays it for two seeds, and both equal the CPU full adjoint (validate_adjoint_split_gpu.jl has the checks). A V100 job runs the same library
# on real CuArrays.
#   julia validate_adjoint_split_cuda_cpu.jl [kernel names] [--sabotage]      SPLITC_SHARD=k/N runs every N-th kernel
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))
isdefined(@__MODULE__, :sab_read) || include(joinpath(@__DIR__, "sabotage_run.jl"))
using Random, LinearAlgebra
include(joinpath(@__DIR__, "validate_adjoint_split_gpu.jl"))

module SCStub
    const CuArray = Array
    module CUDA
        macro allowscalar(e) esc(e) end
        macro atomic(e) esc(e) end
    end
    const bi = Ref(1); const bd = Ref(1); const ti = Ref(1)
    blockIdx() = (x = bi[],)
    blockDim() = (x = bd[],)
    threadIdx() = (x = ti[],)
    macro cuda(a, b, call)
        kv = Dict{Symbol,Any}(x.args[1] => x.args[2] for x in (a, b))
        return esc(quote
            let __T = $(kv[:threads]), __B = $(kv[:blocks])
                for __b in 1:__B, __t in 1:__T
                    SCStub.bi[] = __b; SCStub.ti[] = __t; SCStub.bd[] = __T
                    $call
                end
            end
        end)
    end
end

# the passes of stade_gpu that need CUDA hardware are off
scc_flags() = (warp_atomics = false, sync_free_reduce = false, reduce_block = false, shared_private = false, contraction = 0, split_reduce = 0,
               row_block = 0, col_block = 0, cublas = 0)
scc_env() = sg_cuda_env(; mod = SCStub, wrap = x -> copy(x), tohost = x -> copy(x), flags = scc_flags())

function scc_unit_checks(say)
    src = STADE.io_read_corpus_entry(joinpath(@__DIR__, "split-corpus", "sp_restore.jl"))
    r = STADE.stade_adjoint_split(src)
    pi_ = STADE.stade_gpu(r.initstacks, STADE.cgen_backend_cuda(); scc_flags()...)
    im = Dict{Symbol,Expr}(pi_.host.args[1].args[1] => pi_.host)
    pf = STADE.stade_gpu(r.forward, STADE.cgen_backend_cuda(); scc_flags()..., init_hosts = im)
    say(isempty(pf.dead_stacks) && isempty(pf.compact_stacks), "CUDA: the forward half drops and compacts no stack")
    say(any(k -> occursin("x_stack", STADE.io_expr_to_source(k)), pf.kernels) || occursin("x_stack[", STADE.io_expr_to_source(pf.host)), "CUDA: the forward half still stores the tape")
    pb = STADE.stade_gpu(r.backward, STADE.cgen_backend_cuda(); scc_flags()..., init_hosts = im)
    say(isempty(pb.dead_stacks) && isempty(pb.compact_stacks), "CUDA: the backward half drops and compacts no stack")
end

function validate_adjoint_split_cuda_cpu(; verbose = true, only = String[], sabotage = false, shard = get(ENV, "SPLITC_SHARD", ""))
    checks = 0; bad = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1; verbose && println("NOT OK ", msg)))
    scc_unit_checks(say)
    root = @__DIR__
    counts = Dict{Symbol,Int}()
    k = 0
    sk, sn = isempty(shard) ? (1, 1) : Tuple(parse.(Int, split(shard, "/")))
    for d in sort(filter(d -> endswith(d, "-corpus") && isdir(joinpath(root, d)), readdir(root))), f in sort(readdir(joinpath(root, d)))
        endswith(f, ".jl") || continue
        nm = f[1:end-3]
        isempty(only) || nm in only || continue
        primal = try STADE.io_read_corpus_entry(joinpath(root, d, f)) catch; continue end
        try STADE.parse_kernel(primal) catch; continue end
        k += 1
        (k - 1) % sn + 1 == sk || continue
        Random.seed!(20261122)
        r = try sg_kernel(joinpath(root, d, f); env = scc_env()) catch e; (:error, first(split(sprint(showerror, e), "\n")), NaN, 0) end
        counts[r[1]] = get(counts, r[1], 0) + 1
        # sp_branch_off never takes its branch on a usual draw, so the full adjoint writes nothing there; sp_branch_on is its working twin
        good = r[1] in (:ok, :refused, :host_stacks, :nodraw) || (r[1] == :VACUOUS && nm == "sp_branch_off")
        say(good, "$d/$nm: $(r[1]) $(r[2]) err=$(r[3]) changed=$(r[4])")
    end
    verbose && println("validate_adjoint_split_cuda_cpu: ", join(("$k=$v" for (k, v) in sort(collect(counts))), " "))
    verbose && println("validate_adjoint_split_cuda_cpu: $(checks - bad)/$checks ok")
    if sabotage
        text = read(joinpath(dirname(@__DIR__), "src", "STADE.jl"), String)
        cases = [
            ("dead pops runs on the forward half (CUDA)", "dead_scalars && !(split_half in (:fwd, :fixedpoint)) && (expr = cgen_dead_pops(expr, init_hosts))", "dead_scalars && (expr = cgen_dead_pops(expr, init_hosts))", ["sp_restore"]),
            ("the CUDA stage shrinks the stacks of a half", "dead_stack >= 1 && split_half === nothing && ((host, kernels, dead_stacks", "dead_stack >= 1 && ((host, kernels, dead_stacks", ["sp_restore"]),
            ("a half is not recognised", "endswith(name, \"_b_fwd\") && return :fwd", "endswith(name, \"_b_fwdx\") && return :fwd", ["sp_restore", "sp_carry"]),
        ]
        mktempdir() do tmp
            for (label, old, new, kset) in cases
                isempty(get(ENV, "SPC_SAB_ONLY", "")) || any(t -> occursin(t, label), split(ENV["SPC_SAB_ONLY"], "|")) || continue
                n = length(findall(old, text))
                if n != 1
                    bad += 1; verbose && println("NOT OK sabotage \"$label\": anchor found $n times"); continue
                end
                get(ENV, "SAB_ANCHORS_ONLY", "") == "1" && continue
                path = joinpath(tmp, "STADE_sab.jl"); write(path, replace(text, old => new; count = 1))
                script = """
                    include($(repr(path)))
                    cd($(repr(@__DIR__)))
                    include($(repr(joinpath(@__DIR__, "validate_adjoint_split_cuda_cpu.jl"))))
                    println("RESULT ", validate_adjoint_split_cuda_cpu(; verbose = false, only = $(repr(kset))) ? 0 : 1)
                """
                out = sab_read(script)
                m = match(r"RESULT (\d)", out)
                caught = out == "ERR" || m === nothing || m.captures[1] == "1"
                caught || (bad += 1)
                verbose && println(caught ? "ok     " : "NOT OK ", "sabotage \"$label\": ", caught ? "a check fails" : "NOT CAUGHT")
            end
        end
    end
    return bad == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_adjoint_split_cuda_cpu(; only = String[a for a in ARGS if !startswith(a, "--")], sabotage = "--sabotage" in ARGS) ? 0 : 1)
end
