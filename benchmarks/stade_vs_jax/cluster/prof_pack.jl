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
# The kernel table of nsys: the last column (Name) holds the long C++ name, with commas inside quotes. A cut through the quotes would make the
# following rows unreadable. So the name is cut alone (to w characters) and written again with its quotes.
function kern_lines(txt, w, n)
    L = split(txt, "\n"); i = findfirst(l -> startswith(l, "Time (%)"), L)
    i === nothing && return ""
    out = String[string(L[i])]
    for l in L[i + 1:end]
        length(out) > n && break
        isempty(l) && continue
        c = 0; k = 0
        for (j, ch) in pairs(l)
            ch == ',' && (c += 1)
            if c == 8; k = j; break; end
        end
        k == 0 && continue
        nm = replace(strip(l[nextind(l, k):end], '"'), '"' => '\'')
        push!(out, l[1:k] * "\"" * (length(nm) > w ? first(nm, w) : nm) * "\"")
    end
    return join(out, "\n")
end
# the fields of one CSV line with quotes
function csvfields(l)
    f = String[]; b = IOBuffer(); q = false
    for ch in l
        if ch == '"'; q = !q
        elseif ch == ',' && !q; push!(f, String(take!(b)))
        else write(b, ch); end
    end
    push!(f, String(take!(b))); return f
end
const UNIT = Dict("second" => 1e9, "msecond" => 1e6, "usecond" => 1e3, "nsecond" => 1.0, "byte" => 1.0, "Kbyte" => 1e3, "Mbyte" => 1e6, "Gbyte" => 1e9,
                  "Kibyte" => 1024.0, "Mibyte" => 1048576.0, "Gibyte" => 1073741824.0)
# The ncu log with the details page has one row per kernel and metric. This gives one CSV line per kernel: ID, Kernel Name, one column per metric.
# Times are in ns and sizes are in bytes (the unit of each value is converted with the column Metric Unit).
function ncu_lines(txt, w)
    L = split(txt, "\n"); i = findfirst(l -> startswith(l, "\"ID\""), L)
    i === nothing && return ""
    h = csvfields(L[i]); ix(nm) = findfirst(==(nm), h)
    ci, ck, cm, cu, cv = ix("ID"), ix("Kernel Name"), ix("Metric Name"), ix("Metric Unit"), ix("Metric Value")
    any(isnothing, (ci, ck, cm, cv)) && return ""
    ids = String[]; name = Dict{String,String}(); val = Dict{String,Dict{String,String}}(); mets = String[]
    for l in L[i + 1:end]
        isempty(l) && continue
        f = csvfields(l); length(f) < cv && continue
        all(isdigit, f[ci]) || continue
        id = f[ci]; m = f[cm]
        haskey(name, id) || (push!(ids, id); name[id] = f[ck]; val[id] = Dict{String,String}())
        m in mets || push!(mets, m)
        v = replace(f[cv], "," => ""); u = cu === nothing ? "" : f[cu]
        x = tryparse(Float64, v)
        val[id][m] = x === nothing ? "" : string(x * get(UNIT, u, 1.0))
    end
    out = String["ID,Kernel Name," * join(mets, ",")]
    for id in ids
        nm = replace(name[id], '"' => '\''); nm = length(nm) > w ? first(nm, w) : nm
        push!(out, id * ",\"" * nm * "\"," * join((get(val[id], m, "") for m in mets), ","))
    end
    return join(out, "\n")
end
function main()
    mode, d, capture, pec = ARGS
    ol = filter(l -> startswith(l, "{"), split(rd(joinpath(d, "prog.out")), "\n"))
    prog = isempty(ol) ? "null" : ol[end]
    err = rd(joinpath(d, "prog.err")); err = length(err) > 600 ? last(err, 600) : err
    kern = kern_lines(rd(joinpath(d, "kern.csv")), 160, 400)
    api = lines(rd(joinpath(d, "api.csv")), l -> startswith(l, "Time (%)"), 200, 30)
    nraw = rd(joinpath(d, "ncu.csv")); ncu = ncu_lines(nraw, 160)
    nhead = isempty(ncu) ? join((length(l) > 300 ? first(l, 300) : l for l in first(split(nraw, "\n"), 12)), "\n") : ""   # help to find the format if the table is empty
    println("{\"mode\":" * esc(mode) * ",\"capture\":" * esc(capture) * ",\"prog_exit\":" * pec * ",\"prog\":" * prog * ",\"prog_err\":" * esc(err) *
            ",\"kern_csv\":" * esc(kern) * ",\"api_csv\":" * esc(api) * ",\"ncu_csv\":" * esc(ncu) * ",\"ncu_head\":" * esc(nhead) * "}")
end
main()
