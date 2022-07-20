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

### Geoweighting Examples

```julia
import Microweight as mw
using Statistics

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

# create a small test problem using built-in information
h = 100  # number of households
s = 8  # number of states, regions, etc.
k = 4 # number of characteristics each household has
# the function mtp (make test problem) will create a random problems with these characteristics
tp = mw.mtp(h, s, k)

# explore what's in tp
fieldnames(typeof(tp))
tp.wh
tp.xmat
tp.geotargets


```


## Reweighting



