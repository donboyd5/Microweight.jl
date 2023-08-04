using Revise
import Microweight as mw  # Revise doesn't work for changes to type definitions

using Statistics
# using LineSearches

# using Optimization
# using NLopt
# using Optim
# using OptimizationMOI, Ipopt
# using ModelingToolkit
# using Optimisers


# for Ipopt
# import LinearAlgebra, OpenBLAS32_jll
# LinearAlgebra.BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path)
# also https://docs.juliahub.com/StandaloneIpopt/QHju1/0.4.1/

# https://julianlsolvers.github.io/Optim.jl/latest/#

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
h = 10  # number of households 100
k = 2 # number of characteristics each household has 4

h = 100  # number of households 100
k = 4 # number of characteristics each household has 4

h = 1000  # number of households 100
k = 6 # number of characteristics each household has 4

h = 10000  # number of households 100
k = 20 # number of characteristics each household has 4

h = 100_000  # number of households 100
k = 50 # number of characteristics each household has 4

h = 300_000  # number of households 100
k = 100 # number of characteristics each household has 4

# the function mtp (make test problem) will create a random problem with these characteristics
tp = mw.mtprw(h, k, pctzero=0.3);
fieldnames(typeof(tp))

function qpdiffs(ratio)
  rwtargets_calc = tp.xmat' * (ratio .* tp.wh)
  targpdiffs = (rwtargets_calc .- tp.rwtargets) ./ tp.rwtargets 
  quantile(targpdiffs)
end

# LBFGS seems to be best when ratio error is most important, CCSAQ when target error is most important
algs = ["LD_CCSAQ", "LD_LBFGS", "LD_MMA", "LD_VAR1", "LD_VAR2", "LD_TNEWTON", "LD_TNEWTON_RESTART", "LD_TNEWTON_PRECOND_RESTART", "LD_TNEWTON_PRECOND"]

import Microweight as mw  
res= mw.rwsolve(tp, approach=:minerr, print_interval=10);

res= mw.rwsolve(tp, approach=:minerr);
res= mw.rwsolve(tp, approach=:minerr, method="LD_LBFGS");
res= mw.rwsolve(tp, approach=:minerr, method="LD_CCSAQ");
res= mw.rwsolve(tp, approach=:minerr, method="LD_LBFGS", lb=.2, ub=2.0);
res= mw.rwsolve(tp, approach=:minerr, method="LD_LBFGS", lb=.2, ub=2.0, maxiters=2000);
res= mw.rwsolve(tp, approach=:minerr, method="LD_LBFGS", lb=.1, ub=10.0, rweight=0.0001, maxiters=2000, print_interval=100);
res= mw.rwsolve(tp, approach=:minerr, method="LD_CCSAQ", lb=.1, ub=10.0, rweight=0.0001, maxiters=2000, print_interval=100);
res= mw.rwsolve(tp, approach=:minerr, method=algs[8]);

fieldnames(typeof(res))

res.solve_time
res.objective
quantile(res.u)

qpdiffs(ones(tp.h))
qpdiffs(res.u)



res2 = mw.rwsolve(tp, approach=:minerr, method="spg", lb=.1, ub=10.0, rweight=1e-5)
res2 = mw.rwsolve(tp, approach=:minerr, method="spg", lb=.5, ub=1.5, rweight=0.0)
fieldnames(typeof(res2))
res2.f
# res2.x
quantile(res2.x)
qpdiffs(res2.x)

mw.rwsolve(tp, approach=:minerr, method="xyz")

res3 = mw.rwsolve(tp, approach=:constrain)

res3 = mw.rwsolve(tp, approach=:constrain, lb=.1, ub=10.0, constol=.01)

results = fieldnames(typeof(res3))

res3.objective
quantile(res3.solution)
qpdiffs(res3.solution)


mw.rwsolve(tp, approach=:something)


# only good to here ....
