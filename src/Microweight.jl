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

##############################################################################
##
## Dependencies
##
##############################################################################

using LsqFit

##############################################################################
##
## Exported methods and types
##
##############################################################################

# order these alphabetically by file
export
  # src\functions_poisson_typestable.jl
  geo_targets, geo_weights, objfn, sspd, lsq, objvec,
  # src\functions_poisson_fg_typestable.jl
  # src\make_test_problems.jl
  mtp,
  # src\get_taxdata_problems.jl
  get_taxprob,
  GeoweightProblem

##############################################################################
##
## Load files
##
##############################################################################

# files in src
include("functions_poisson_typestable.jl")
include("functions_poisson_fg_typestable.jl")
include("make_test_problems.jl")
include("get_taxdata_problems.jl")

include("functions_test.jl")
include("scaling.jl")
include("types.jl")

end
