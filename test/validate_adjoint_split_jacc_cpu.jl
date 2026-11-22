# Regression for v0.5.35: the JACC code of the two halves of stade_adjoint_split, run on the CPU with a stand-in JACC (a serial `for` loop per launch,
# the one of validate_jacc_glue.jl; the hardware passes are off). The forward half records the tape, the backward half replays it for two seeds, and both
# equal the CPU full adjoint (validate_adjoint_split_gpu.jl has the checks). A V100 job runs the same library on real arrays.
#   The defect this guards: the GPU stage sees one function. The forward half only writes its tape, so cgen_dead_pops deleted the stores nothing in the
#   function reads, and the backward half restored zeros (NaN in 1 / sqrt(s); every restore of an array wrong). The stage now leaves a half alone.
#   julia validate_adjoint_split_jacc_cpu.jl [kernel names] [--sabotage]      SPLITJ_SHARD=k/N runs every N-th kernel
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))
isdefined(@__MODULE__, :sab_read) || include(joinpath(@__DIR__, "sabotage_run.jl"))
using Random, LinearAlgebra
include(joinpath(@__DIR__, "validate_adjoint_split_gpu.jl"))

module SJStub
    module JACC
        array(x) = copy(x)
        to_host(x) = copy(x)
        synchronize() = nothing
        zeros(T, d...) = Base.zeros(T, d...)
        macro parallel_for(a...)
            call = a[end]
            rng = nothing
            for x in a[1:end-1]
                x isa Expr && x.head === :(=) && x.args[1] === :range && (rng = x.args[2])
            end
            rng === nothing && error("stand-in parallel_for: no range")
            return esc(:(for __ji in 1:($rng); $(call.args[1])(__ji, $(call.args[2:end]...)); end))
        end
        macro parallel_reduce(a...) return :(error("stand-in JACC: parallel_reduce was reached")) end
    end
    module CUDA
        macro allowscalar(e) esc(e) end
    end
    module Atomix
        macro atomic(e) esc(e) end
    end
end

spj_env() = (mod = SJStub, wrap = x -> copy(x), tohost = x -> copy(x),
             flags = (warp_atomics = false, split_reduce = 0, jacc_contraction = 0, reduction_threshold = typemax(Int) ÷ 2, glue_kernels = true),
             atomic_macro = Expr(:., :Atomix, QuoteNode(Symbol("@atomic"))))

# plan-level unit: the forward half keeps its tape stores in both stages, and a half keeps the stack of the other (no shrink) in the CUDA stage
function spj_unit_checks(say)
    src = STADE.io_read_corpus_entry(joinpath(@__DIR__, "split-corpus", "sp_restore.jl"))
    r = STADE.stade_adjoint_split(src)
    im = Dict{Symbol,Expr}(r.initstacks.args[1].args[1] => r.initstacks)
    pj = STADE.stade_jacc(r.forward; init_hosts = im)
    say(any(k -> occursin("x_stack", STADE.io_expr_to_source(k)), pj.kernels) || occursin("x_stack[", STADE.io_expr_to_source(pj.host)), "JACC: the forward half still stores the tape")
    pc = STADE.stade_gpu(r.forward, STADE.cgen_backend_cuda(); init_hosts = im)
    say(isempty(pc.dead_stacks) && isempty(pc.compact_stacks), "CUDA: the forward half drops and compacts no stack")
    say(any(k -> occursin("x_stack", STADE.io_expr_to_source(k)), pc.kernels) || occursin("x_stack[", STADE.io_expr_to_source(pc.host)), "CUDA: the forward half still stores the tape")
    pb = STADE.stade_gpu(r.backward, STADE.cgen_backend_cuda(); init_hosts = im)
    say(isempty(pb.dead_stacks) && isempty(pb.compact_stacks), "CUDA: the backward half drops and compacts no stack")
    # a whole adjoint is untouched: the stage still drops what nothing reads
    say(STADE.cgen_split_half(r.forward) === :fwd && STADE.cgen_split_half(r.backward) === :bwd && STADE.cgen_split_half(Meta.parse("function f_b(x)\nreturn nothing\nend")) === nothing, "cgen_split_half names the two halves only")
end

function validate_adjoint_split_jacc_cpu(; verbose = true, only = String[], sabotage = false, shard = get(ENV, "SPLITJ_SHARD", ""))
    checks = 0; bad = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1; verbose && println("NOT OK ", msg)))
    spj_unit_checks(say)
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
        r = try sg_kernel(joinpath(root, d, f); env = spj_env()) catch e; (:error, first(split(sprint(showerror, e), "\n")), NaN, 0) end
        counts[r[1]] = get(counts, r[1], 0) + 1
        # sp_branch_off never takes its branch on a usual draw, so the full adjoint writes nothing there; sp_branch_on is its working twin
        good = r[1] in (:ok, :refused, :host_stacks, :nodraw) || (r[1] == :VACUOUS && nm == "sp_branch_off")
        # the stand-in has no block reduction: a kernel whose loop is long enough to launch one cannot run here (the V100 job covers it)
        good |= r[1] == :error && occursin("parallel_reduce was reached", r[2])
        say(good, "$d/$nm: $(r[1]) $(r[2]) err=$(r[3]) changed=$(r[4])")
    end
    verbose && println("validate_adjoint_split_jacc_cpu: ", join(("$k=$v" for (k, v) in sort(collect(counts))), " "))
    verbose && println("validate_adjoint_split_jacc_cpu: $(checks - bad)/$checks ok")
    if sabotage
        text = read(joinpath(dirname(@__DIR__), "src", "STADE.jl"), String)
        cases = [
            ("dead pops runs on the forward half (JACC)", "dead_scalars && jacc_dead_pops && cgen_split_half(expr) !== :fwd && (expr = cgen_dead_pops(expr, init_hosts))", "dead_scalars && jacc_dead_pops && (expr = cgen_dead_pops(expr, init_hosts))", ["sp_restore", "sp_carry"]),
            ("dead pops runs on the forward half (CUDA)", "dead_scalars && split_half !== :fwd && (expr = cgen_dead_pops(expr, init_hosts))", "dead_scalars && (expr = cgen_dead_pops(expr, init_hosts))", ["sp_restore"]),
            ("the CUDA stage shrinks the stacks of a half", "dead_stack >= 1 && split_half === nothing && ((host, kernels, dead_stacks", "dead_stack >= 1 && ((host, kernels, dead_stacks", ["sp_restore"]),
            ("a half is not recognised", "endswith(name, \"_b_fwd\") && return :fwd", "endswith(name, \"_b_fwdx\") && return :fwd", ["sp_restore", "sp_carry"]),
        ]
        mktempdir() do tmp
            for (label, old, new, kset) in cases
                isempty(get(ENV, "SPJ_SAB_ONLY", "")) || any(t -> occursin(t, label), split(ENV["SPJ_SAB_ONLY"], "|")) || continue
                n = length(findall(old, text))
                if n != 1
                    bad += 1; verbose && println("NOT OK sabotage \"$label\": anchor found $n times"); continue
                end
                get(ENV, "SAB_ANCHORS_ONLY", "") == "1" && continue
                path = joinpath(tmp, "STADE_sab.jl"); write(path, replace(text, old => new; count = 1))
                script = """
                    include($(repr(path)))
                    cd($(repr(@__DIR__)))
                    include($(repr(joinpath(@__DIR__, "validate_adjoint_split_jacc_cpu.jl"))))
                    println("RESULT ", validate_adjoint_split_jacc_cpu(; verbose = false, only = $(repr(kset))) ? 0 : 1)
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
    exit(validate_adjoint_split_jacc_cpu(; only = String[a for a in ARGS if !startswith(a, "--")], sabotage = "--sabotage" in ARGS) ? 0 : 1)
end
