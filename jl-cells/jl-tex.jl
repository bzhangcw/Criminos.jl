# LaTeX label definitions for policy names
# Matches macros from paper-informs/macro.tex

using LaTeXStrings

# Policy labels (matching \polnul, \pollow, \polhigh, \polsd)
TEX_POL_NULL = L"\texttt{null}"
TEX_POL_LOW = L"\texttt{low-risk}"
TEX_POL_HIGH = L"\texttt{high-risk}"
TEX_POL_SD = L"\texttt{eo}"

# Dictionary for easy lookup
TEX_POLICY_LABELS = Dict(
    "null" => TEX_POL_NULL,
    "low-risk" => TEX_POL_LOW,
    "high-risk" => TEX_POL_HIGH,
    "steady-state" => TEX_POL_SD,
)

# Helper function to get policy label
get_tex_label(policy::String) = get(TEX_POLICY_LABELS, policy, policy)
