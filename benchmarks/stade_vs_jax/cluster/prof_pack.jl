# Packs the files of one profiled run (written by prof_cmd.sh) into ONE JSON line on the standard output.
# Usage: julia prof_pack.jl MODE DIR CAPTURE EXIT_CODE_OF_THE_PROGRAM
function esc(s)
    io = IOBuffer(); write(io, '"')
    for c in String(s)
        c == '"' ? write(io, "\\\"") : c == '\\' ? write(io, "\\\\") : c == '\n' ? write(io, "\\n") : (c < ' ' ? write(io, ' ') : write(io, c))
    end
    write(io, '"'); return String(take!(io))
end
rd(p) = isfile(p) ? read(p, String) : ""
# the lines of a text from the first line that satisfies start, each cut to w characters, at most n lines
function lines(txt, start, w, n)
    L = split(txt, "\n"); i = findfirst(start, L)
    i === nothing && return ""
    return join((length(l) > w ? first(l, w) : l for l in L[i:min(end, i + n - 1)] if !isempty(l)), "\n")
end
function main()
    mode, d, capture, pec = ARGS
    ol = filter(l -> startswith(l, "{"), split(rd(joinpath(d, "prog.out")), "\n"))
    prog = isempty(ol) ? "null" : ol[end]
    err = rd(joinpath(d, "prog.err")); err = length(err) > 600 ? last(err, 600) : err
    kern = lines(rd(joinpath(d, "kern.csv")), l -> startswith(l, "Time (%)"), 330, 400)
    api = lines(rd(joinpath(d, "api.csv")), l -> startswith(l, "Time (%)"), 200, 30)
    ncu = lines(rd(joinpath(d, "ncu.csv")), l -> startswith(l, "\"ID\""), 900, 400)
    println("{\"mode\":" * esc(mode) * ",\"capture\":" * esc(capture) * ",\"prog_exit\":" * pec * ",\"prog\":" * prog * ",\"prog_err\":" * esc(err) *
            ",\"kern_csv\":" * esc(kern) * ",\"api_csv\":" * esc(api) * ",\"ncu_csv\":" * esc(ncu) * "}")
end
main()
