# Microweight.jl

Microweight.jl has functions that allow the user to adjust weights for microdata
files including social science surveys such as the *American Community Survey*,
public use microdata files of pseudo tax returns, and other similar files.

It has two main goals: reweighting, and geo-weighting.

## Installation

Assuming that you already have Julia correctly installed, it suffices to import
Microweight.jl in the standard way:

```julia
import Pkg; Pkg.add("Microweight")
```

In addition, it is helpful to import Statistics to examine resuls.

The package LineSearches is needed for some algorithmic options. (Change this.)

## Geoweighting


See the examples/geoweighting directory. The file names should be self-explanatory. There are examples that:

+   [Solve a simple problem multiple ways](https://github.com/donboyd5/Microweight.jl/blob/main/examples/geoweighting/solve_simple_problem.jl)
+   Create a problem using ACS data, and then solve it


