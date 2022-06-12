module Microweight

# using Revise  # in REPL
using LsqFit

export geo_targets, geo_weights, objfn, sspd, lsq, objvec,
    mtp,
    get_taxprob

include("functions_poisson_typestable.jl")
include("functions_poisson_fg_typestable.jl")
include("make_test_problems.jl")
include("get_taxdata_problems.jl")

include("functions_test.jl")

end
