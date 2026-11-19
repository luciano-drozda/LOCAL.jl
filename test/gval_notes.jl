# Reporting helpers shared by validate_corpus_gpu.jl and validate_large_draw.jl (the first runs on include, so the second cannot include it).
# A draw asks for integers of at least int_lo; the baseline generator halves the upper end of the range after a candidate whose output is too large
# or diverges, so a run that asked for a large draw can come back with small integers. The row says so instead of passing as a large one.
function gval_narrowed_note(int_args::Dict, pinned::Dict, int_lo::Int)
    low = sort([String(a) for (a, v) in int_args if !haskey(pinned, a) && v < int_lo])
    return isempty(low) ? "" : " narrowed=" * join(low, ",") * "<" * string(int_lo)
end

