module Microweight

# using Revise  # in REPL

# export f, g
export geo_targets, geo_weights, objfn, sspd

include("functions_poisson.jl")
include("functions_poisson_fg.jl")

end
