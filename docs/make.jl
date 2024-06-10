using Documenter
using Criminos

makedocs(
    sitename="Criminos.jl",
    authors="Chuwen Zhang, Pengyi Shi, Amy Ward",
    format=Documenter.HTML(
        # See  https://github.com/JuliaDocs/Documenter.jl/issues/868
        prettyurls=get(ENV, "CI", nothing) == "true",
        collapselevel=1,
    ),
    clean=true,
    pages=[
        "Home" => "index.md",
        # "Tutorials" => Any[
        #     "Basic"=>list_of_sorted_files("tutorial/basic", TUTORIAL_BASIC_DIR),
        #     "Advanced"=>list_of_sorted_files(
        #         "tutorial/advanced",
        #         TUTORIAL_ADVANCED_DIR,
        #     ),
        #     "Theory"=>list_of_sorted_files(
        #         "tutorial/theory",
        #         TUTORIAL_THEORY_DIR,
        #     ),
        # ],
        # "How-to guides" => list_of_sorted_files("guides", GUIDES_DIR),
        # "Examples" => list_of_sorted_files("examples", EXAMPLES_DIR),
        "Basic" => "basic.md",
        "Dynamic τ" => "decision.md",
        "API" => "api.md",
        "Data Source" => "data.md",
        # "Release notes" => "release_notes.md",
    ],
    doctestfilters=[r"[\s\-]?\d\.\d{6}e[\+\-]\d{2}"],
    # modules=[DRSOM]
)


deploydocs(
    repo="https://github.com/bzhangcw/Criminos.jl.git",
    push_preview=true,
    devbranch="dev",
    devurl="dev",
    versions=["stable" => "v^", "v#.#", "dev" => "dev"],
)