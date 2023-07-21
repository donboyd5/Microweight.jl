__precompile__()

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

#= TODO
- memory
- kwargs for inner methods
- improve Optimization.Optimizers https://optimization.sciml.ai/stable/optimization_packages/optimisers/
    bound constraints or penalty for direct method?
    how adjust learning rates?
- show function call
- ACS data
- examples
- call from R, call from python
- documentation

- explore https://github.com/emmt/OptimPackNextGen.jl -- vmlmb
  https://github.com/emmt/OptimPackNextGen.jl/blob/master/doc/quasinewton.md
  seems to be like LBFGS
  see https://github.com/emmt/OptimPackNextGen.jl/blob/master/test/rosenbrock.jl lines 81+
  consider for both poisson and direct
  note that it does scaling
  https://github.com/emmt/OptimPackNextGen.jl/issues/8

- explore nlboxsolve jfnk
  https://juliahub.com/ui/Packages/NLboxsolve/bk0LI/0.4.2
  https://github.com/RJDennis/NLboxsolve.jl
  The key elements to a problem are
    a vector-function containing the system of equations to be solved F(x)
    an initial guess at the solution, x (1d-array)
    the lower, lb (1d-array with default enteries equaling -Inf), and upper, ub (1d-array with default enteries equaling Inf)
    bounds that form the box-constraint.
    plus...
  soln = nlboxsolve(F,x,l,u,xtol=1e-10,ftol=1e-10,maxiters=200,method=:jfnk,sparsejac=:yes,krylovdim=20)
  see https://github.com/RJDennis/NLboxsolve.jl/blob/main/src/boxsolvers.jl line 4561
  if method == :jfnk
        return constrained_jacobian_free_newton_krylov(f,x,lb,ub,xtol=xtol,ftol=ftol,maxiters=maxiters)

- tests

https://juliasmoothoptimizers.github.io/DCISolver.jl/stable/example/
https://github.com/JuliaSmoothOptimizers/NLPModels.jl
https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl


DONE:
- beta scaling BAD, don't do it
- figure out krylov why do callback results look good but final results do not???  DONE

=#

module Microweight

##############################################################################
##
## Dependencies
##
##############################################################################

using Parameters, Printf, Statistics
# optimization helpers
using LinearAlgebra, ChainRules
using ForwardDiff, LineSearches, NLSolversBase, FiniteDiff, ReverseDiff, Zygote
# using ModelingToolkit
using LeastSquaresOptim, LsqFit, MINPACK, NLsolve, Optim
using Optimization, OptimizationNLopt, OptimizationOptimisers, OptimizationOptimJL
# using OptimizationMOI, Ipopt
# using Mads  # haven't figured out how to make it work well
# import Pkg; Pkg.precompile()
# import Pkg; Pkg.add("OptimizationMOI")
# import Pkg; Pkg.add("Ipopt")

##############################################################################
##
## Exported methods and types
##
##############################################################################
# order these alphabetically by file
export mtp, geosolve, get_taxprob, objfn_reweight, rwsolve
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
include("functions_display_and_callbacks.jl")
include("functions_utilities.jl")
include("get_taxdata_problems.jl")
include("make_test_problems.jl")
include("scaling.jl")

# geoweight direct functions and solvers 
include("functions_geoweight_direct.jl")
include("functions_reweight.jl")

include("direct_optz_nlopt.jl")
include("direct_optz_optim.jl")
include("direct_optz_optimisers.jl")

# poisson functions and solvers
include("functions_poisson.jl")
include("functions_poisson_fg.jl")

# include("poisson_krylov.jl")
include("poisson_lsoptim.jl")
include("poisson_lsqlm.jl")
# include("poisson_mads.jl")
include("poisson_minpack_fsolve.jl")
include("poisson_nlsolve.jl")
include("poisson_optz_nlopt.jl")
include("poisson_optz_optim.jl")
include("poisson_optz_optimisers.jl")

# reoweight direct functions and solvers 
# include("functions_reweight_direct.jl")

# include("reweight_optz_nlopt.jl")
include("reweight_optz_optim.jl")

# functions underlying all calculations


# misc

# testing and play
include("experimental/functions_test.jl")

end
