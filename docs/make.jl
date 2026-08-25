using Documenter
using CCOpt

makedocs(
    sitename = "CCOpt.jl",
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.KaTeX()
    ),
    modules = [CCOpt],
    repo = "https://github.com/MadNLP/CCOpt.jl/blob/{commit}{path}#{line}",
    checkdocs = :none,
    clean=true,
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
    ]
)

deploydocs(
    repo = "github.com/MadNLP/CCOpt.jl.git",
    target = "build",
    devbranch = "master",
    devurl = "dev",
    push_preview = true,
)
