module Microweight

# using Revise  # in REPL
using LsqFit

# export f, g
export geo_targets, geo_weights, objfn, sspd, lsq

include("functions_poisson.jl")
include("functions_poisson_fg.jl")

function lsq(ibeta, wh, xmat, geotargets)
  LsqFit.lmfit(f, ibeta, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=50)
end

end
