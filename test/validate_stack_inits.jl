# Regression for v0.5.34: a float local that a generated sweep saves on a stack is assigned at function level before its first save.
#   Defect: with a condition no longer a read, a variable read only by a condition (phc_saha_bisect's `gm`) has no snapshot site, but the sweep over its loop still
#   saves it at a block boundary. Its `gm = 0.0` came from the sites, so it was gone: the save raised an UndefVarError when the loop ran zero times, and in the
#   GPU glue code where the loop body is a kernel. A CPU run hid it, since Julia gives the variable function scope once a later top-level statement assigns it.
#   Fix: agen_local_primal_inits also initialises every float local that has a value stack.
# Reads generated code only (adjoint and HVP, both keep_push_pop values, fuse_ii_loops on and off) for every kernel of every corpus under test/.
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))

# a save of `v` onto a stack: `push!(<stack>, v)` or `<stack>[<index>] = v`
si_save_re(v) = Regex("^\\s*(push!\\(\\w*stack\\w*, $v\\)|\\w*stack\\w*\\[.*\\] = $v)\\s*\$")

# the saved locals of one generated function that no function-level assignment precedes
function si_missing(kernel, src::String)
    lines = split(src, "\n")
    out = Symbol[]
    for (v, k) in kernel.sig.kinds
        (k == :scalar_float && !(v in kernel.sig.args)) || continue
        first_save = findfirst(l -> occursin(si_save_re(v), l), lines)
        first_save === nothing && continue
        init = findfirst(l -> occursin(Regex("^    $v = "), l), lines)
        (init === nothing || init > first_save) && push!(out, v)
    end
    return sort(out)
end

function validate_stack_inits(; verbose = true, only = String[], corpora = nothing)
    bad = 0; checks = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1; verbose && println("NOT OK ", msg)))
    root = @__DIR__
    dirs = corpora === nothing ? sort(filter(d -> endswith(d, "-corpus") && isdir(joinpath(root, d)), readdir(root))) : corpora
    saved_locals = 0
    for d in dirs, f in sort(readdir(joinpath(root, d)))
        endswith(f, ".jl") || continue
        nm = f[1:end-3]
        isempty(only) || nm in only || continue
        primal = try STADE.io_read_corpus_entry(joinpath(root, d, f)) catch; continue end
        for kp in (false, true), fz in (false, true), mode in (:adjoint, :hvp)
            g = try
                mode == :adjoint ? STADE.stade_adjoint(primal; keep_push_pop = kp, fuse_ii_loops = fz).adjoint :
                                   STADE.stade_hvp(primal; keep_push_pop = kp, fuse_ii_loops = fz).hvp
            catch
                continue   # a kernel a mode refuses is not this suite's concern
            end
            kernel = STADE.parse_kernel(primal)
            miss = si_missing(kernel, string(g))
            say(isempty(miss), "$d/$nm $mode keep_push_pop=$kp fuse_ii_loops=$fz: saved without an initial value: $miss")
            saved_locals += count(v -> kernel.sig.kinds[v] == :scalar_float && !(v in kernel.sig.args) && occursin(r"stack", string(g)), keys(kernel.sig.kinds))
        end
    end
    # the check must see saves at all: a regex that matched nothing would pass everything
    say(saved_locals > 0 || !isempty(only), "no kernel had a float local with a stack: the save pattern matches nothing")
    verbose && println("validate_stack_inits: $(checks - bad)/$checks ok")
    return bad == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_stack_inits(; only = String[a for a in ARGS if !startswith(a, "--")]) ? 0 : 1)
end
