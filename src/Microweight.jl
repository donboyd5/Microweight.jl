"""
# Microweight.jl
Welcome to Microweight.jl!

Microweight.jl is a package used to construct or adjust weights for microdata
files including social science surveys such as the *American Community Survey*,
public use microdata files of pseudo tax returns, and other similar files.

## REPL help
`?` followed by a method name (`?BFGS`) prints help to the terminal.

## Documentation
Besides the help provided at the REPL, it is possible to find help and general
documentation online at link .
"""
module Microweight


#= TODO:
- scaling
- show function call
- better trace arrangement
- lbgfs
- krylov

https://github.com/JuliaNLSolvers/Optim.jl/blob/master/src/multivariate/solvers/second_order/krylov_trust_region.jl

res5 = Optim.optimize(f, lsres.param, ConjugateGradient(),
Optim.Options(g_tol = 1e-6, iterations = 10, store_trace = true, show_trace = true);
 autodiff = :forward)

res6 = Optim.optimize(f, ibeta, GradientDescent(),
  Optim.Options(g_tol = 1e-6, iterations = 10, store_trace = true, show_trace = true);
  autodiff = :forward) # seems to become very slow as problem size increases

res7 = Optim.optimize(f, ibeta, MomentumGradientDescent(),
  Optim.Options(g_tol = 1e-6, iterations = 10, store_trace = true, show_trace = true);
  autodiff = :forward)
# 1602.8 3.078448e-11

# really good after 3 iterations 562 secs
res8 = Optim.optimize(f, ibeta, AcceleratedGradientDescent(),
  Optim.Options(g_tol = 1e-6, iterations = 10, store_trace = true, show_trace = true);
  autodiff = :forward)

res12 = Optim.optimize(f, g!, ibeta, ConjugateGradient(eta=0.01; alphaguess = LineSearches.InitialConstantChange(), linesearch = LineSearches.HagerZhang()),
  Optim.Options(g_tol = 1e-6, iterations = 1_000, store_trace = true, show_trace = true))
# 4.669833e+03 after 10k
# 2.030909e+03 after 20k
# 1.173882e+03 after 30k
#  after 40k
#  after 50k
res12a = Optim.optimize(f, g!, minimizer(res12a), ConjugateGradient(eta=0.01; alphaguess = LineSearches.InitialConstantChange(), linesearch = LineSearches.HagerZhang()),
  Optim.Options(g_tol = 1e-6, iterations = 10_000, store_trace = true, show_trace = true))

nlboxsolve??


=#

##############################################################################
##
## Dependencies
##
##############################################################################

using Parameters, Printf, Statistics
# optimization helpers
using LineSearches, ForwardDiff, LineSearches, NLSolversBase, FiniteDiff, ReverseDiff, Zygote
using ModelingToolkit
using LeastSquaresOptim, LsqFit, MINPACK, NLsolve, Optim
using Optimization, OptimizationOptimJL, OptimizationNLopt
# using Mads  # haven't figured out how to make it work well


##############################################################################
##
## Exported methods and types
##
##############################################################################

# order these alphabetically by file
export mtp, geosolve, get_taxprob
  # # src\api.jl
  # geosolve,
  # # src\functions_poisson_typestable.jl
  # # geo_targets, geo_weights, objfn, sspd, lsq, objvec,
  # # src\functions_poisson_fg_typestable.jl
  # # src\make_test_problems.jl
  # mtp,
  # # src\get_taxdata_problems.jl
  # get_taxprob,
  # GeoweightProblem

##############################################################################
##
## Load files
##
##############################################################################

# files in src ---------------------------------------------------------------

# types are needed in several functions so load them first
include("types.jl")

include("api.jl") # function to route things

# helpers
include("functions_utilities.jl")
include("make_test_problems.jl")
include("get_taxdata_problems.jl")
include("scaling.jl")

# direct functions and solvers
include("functions_direct.jl")

include("direct_cg.jl")

# poisson functions and solvers
include("functions_poisson.jl")
include("functions_poisson_fg.jl")

include("poisson_cgoptim.jl")
include("poisson_cgoptim2.jl")
include("poisson_lsoptim.jl")
include("poisson_lsqlm.jl")
# include("poisson_mads.jl")
include("poisson_minpack.jl")
include("poisson_newttrust.jl")
include("poisson_krylov.jl")

# functions underlying all calculations


# misc

# testing and play
include("experimental/functions_test.jl")

end
