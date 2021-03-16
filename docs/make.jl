using MemoryBuffers
using Documenter

DocMeta.setdocmeta!(MemoryBuffers, :DocTestSetup, :(using MemoryBuffers); recursive=true)

makedocs(;
    modules=[MemoryBuffers],
    authors="Zachary P. Christensen <zchristensen7@gmail.com> and contributors",
    repo="https://github.com/Tokazama/MemoryBuffers.jl/blob/{commit}{path}#{line}",
    sitename="MemoryBuffers.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Tokazama.github.io/MemoryBuffers.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Tokazama/MemoryBuffers.jl",
)
