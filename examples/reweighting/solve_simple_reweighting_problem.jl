using Revise
import Microweight as mw  # Revise doesn't work for changes to type definitions
using Statistics
using LineSearches

# Create a test problem that has the following characteristics:
#   h households
#   s states (areas, or regions, etc.)
#   k characteristics
#   xmat: an x-matrix of household characteristics, with h rows and k columns
#   wh: national weights of households - a vector with h rows and 1 column
#   geotargets: an s x k matrix of targets, one for each target, for each state

# the above is the minimum set of information needed to solve for:
#   whs: an h x s matrix that has one weight per household (h) per state (s),
#     with the characteristics that:
#        for each household the weights sum to national weights (wh), or as close to that as possible
#        weighed sums of the characteristics for each state, calculated using these weights, equal or are as close as possible
#          to the geotargets

## create a small test problem using built-in information

# small for initial compilation
h = 100  # number of households 100
k = 4 # number of characteristics each household has 4

# the function mtp (make test problem) will create a random problem with these characteristics
tp = mw.mtprw(h, k)
fieldnames(typeof(tp))

tp.h
tp.k
tp.wh
tp.xmat
tp.rwtargets
tp.rwtargets_calc
tp.rwtargets_diff
tp.rwtargets_calc ./ tp.rwtargets .- 1.


tmp = mw.objfn_reweight(
    tp.wh, tp.xmat,
    tp.rwtargets;
    whweight=0.5,
    pow=2.0,
    targstop=true, whstop=true,
    display_progress=true)

# return objval, targdiffs, whdiffs, targstop, whstop
fieldnames(typeof(tmp))
tmp[1]
tmp[2]
tmp[3]



# kwargs must be common options or allowable options for NLopt that Optimization will pass through to NLopt
kwkeys_method = (:maxtime, :abstol, :reltol)
kwkeys_algo = (:stopval, )
kwargs_defaults = Dict(:stopval => 1e-4)
kwargs_use = kwargs_keep(kwargs; kwkeys_method=kwkeys_method, kwkeys_algo=kwkeys_algo, kwargs_defaults=kwargs_defaults)

println("Household weights component weight: ", whweight)

opt = Optimization.solve(fprob, NLopt.eval(algorithm), maxiters=maxiter, callback=cb_direct; kwargs_use...)



