# Census of host stacks in the keep_push_pop = false adjoint, over the corpus (v0.5.26). Plan only, nothing runs on a device.
# A stack the layout cannot give a closed form is "tainted" (Tier B, a loop bound assigned inside an ancestor loop that is itself
# below a ragged block): its push!/pop! stay in the generated code, and on a GPU every element goes through the host.
# For each kernel, with keep_push_pop = false and fuse_ii_loops = true, it prints one line
#   TAINT {"name":..., "offender":..., "tainted":[stack names], "push":n, "pop":n}
# where `offender` is agen_tier_b_offender (the first ragged loop bound, or null) and push/pop count the calls left in the adjoint.
# With CENSUS_HOST=1 it also reports the CUDA host (kernel, glue, push!, pop!, @allowscalar counts).
#   julia census_tainted_stacks.jl [kernel names...]     (CENSUS_SHARD=k/N runs every N-th kernel)

isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

function census_corpus_paths()
    paths = Dict{String,String}()
    for d in sort(filter(isdir, readdir(@__DIR__; join = true))), f in sort(readdir(d; join = true))
        endswith(f, ".jl") && occursin("-corpus", d) || continue
        n = splitext(basename(f))[1]
        occursin(r"_(b|d|hv|bd|cuda|jacc)$", n) && continue
        haskey(paths, n) || (paths[n] = f)
    end
    return paths
end

census_count(t, p) = length(collect(eachmatch(p, t)))

function census_taint_kernel(name, path; host::Bool = false)
    primal = STADE.io_read_corpus_entry(path)
    off = STADE.agen_tier_b_offender(STADE.parse_kernel(primal))
    gen = STADE.stade_adjoint(primal; keep_push_pop = false, fuse_ii_loops = true)
    txt = string(gen.adjoint)
    rec = Dict{String,Any}("name" => name, "offender" => off === nothing ? nothing : string(off),
                           "push" => census_count(txt, r"push!\("), "pop" => census_count(txt, r"pop!\("))
    if host && rec["push"] + rec["pop"] > 0
        pl = STADE.stade_cuda(gen.adjoint); t = string(pl.host)
        rec["host_push"] = census_count(t, r"push!\("); rec["host_pop"] = census_count(t, r"pop!\(")
        rec["allowscalar"] = census_count(t, r"allowscalar"); rec["glue"] = census_count(t, r"cuda_glue_[A-Za-z_0-9]*!")
        rec["par"] = census_count(t, r"cuda_kernel_[A-Za-z_0-9]*!")
    end
    return rec
end

function census_tainted_main(args)
    paths = census_corpus_paths()
    names = isempty(args) ? sort(collect(keys(paths))) : args
    sh = get(ENV, "CENSUS_SHARD", "")
    if !isempty(sh)
        (k, n) = parse.(Int, split(sh, "/"))
        names = [nm for (i, nm) in enumerate(names) if mod(i - 1, n) == k - 1]
    end
    host = get(ENV, "CENSUS_HOST", "") == "1"
    for nm in names
        t0 = time()
        try
            r = census_taint_kernel(nm, paths[nm]; host)
            r["s"] = round(time() - t0; digits = 1)
            println("TAINT ", join(["\"$k\":" * (v === nothing ? "null" : v isa AbstractString ? "\"$v\"" : string(v)) for (k, v) in sort(collect(r); by = first)], ", "))
        catch e
            println("TAINT_ERROR ", nm, " ", first(sprint(showerror, e), 160))
        end
        flush(stdout)
    end
end

abspath(PROGRAM_FILE) == @__FILE__() && census_tainted_main(ARGS)
