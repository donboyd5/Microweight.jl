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
# the function mtp (make test problem) will create a random problem with these characteristics
tp = mw.mtp(h, s, k)

# explore what's in tp
fieldnames(typeof(tp))
# (:wh, :xmat, :geotargets, :h, :k, :s, :target_sums, :target_calcs, :target_diffs, :wh_scaled, :xmat_scaled, :geotargets_scaled)
# wh, xmat, and geotargets are fields you will always need to solve for state weights
# we'll look at other fields later - they're not needed to solve for state weights
tp.wh
tp.xmat
tp.geotargets

# solve the problem with all defaults, which will use poisson approach and lm_lsqfit
# this should be fine for most problems
resp1 = mw.geosolve(tp)
# geosolve prints results of each iteration; we'll discuss they're meaning elsewhere

# explore what's in res
fieldnames(typeof(resp1))
# (:approach, :method, :success, :iterations, :eseconds, :sspd, :beta, :beta0, :shares, :shares0, :whs, :wh_calc, :wh_pdiffs, :wh_pdqtiles, :geotargets_calc, :targ_pdiffs, :targ_pdqtiles, :solver_result, :problem)
resp1.approach # :poisson
resp1.method # :lm_lsqfit
resp1.sspd  # sum of squared percentage differences from the geotargets - a summary measure of how well we did
resp1.eseconds # elapsed seconds (including any needed compilation time)
resp1.whs  # weights for households, for each state -- the key result

resp1.geotargets_calc # s x k matrix of geographic targets calculated using the state weights
# we can compare this to tp.geotargets - our intended target values - we want them to be extremely close
resp1.targ_pdiffs # percentage differences of calculated targets from intended targets
# i.e., this is (resp1.geotargets_calc ./ tp.geotargets .- 1.0) * 100.0
# we'd like all of these to be zero
resp1.targ_pdqtiles # quantiles of target percentage differences, hopefully all near zero

# there are similar results for the calculated state weights - we want each household's sum of state weights to be near its national weight
resp1.wh_pdiffs # percentage differences of household weights summed across states from given national weights
# calculated as (resp1.wh_calc - tp.wh) ./ tp.wh * 100.
resp1.wh_pdqtiles # quantiles of percentage differences of sums of state weights, relative to given national state weights - should be near zero

# solve by a few different methods, sticking with the poisson approach
# why multiple methods? sometimes some algorithms work better than others
# use another poisson method - lm_lsoptim, and state the poisson approach explicity (not necessary, as it is default)
resp2 = mw.geosolve(tp, approach=:poisson, method=:lm_lsoptim)
resp3 = mw.geosolve(tp, approach=:poisson, method=:lm_minpack, objscale=1.0) # investigate other options
# NLOPT and OPTIM not working well - investigate?


# did all three methods have very low sums of squared percentage differences from targets?
resp1.sspd
resp2.sspd
resp3.sspd
# yes; differences across methods reflect different options and stopping criteria and do not necessarily indicate
# that one was better than the other; the real question is whether results are "good enough"
# for example resp2 has the "worst" sspd, but are the state weight-sums close enough to national weights, and are
# calculated targets close enough to intended targets?
resp2.wh_pdqtiles # state-weight sums for each household are close to the household's national weight (all % diffs are near 0)
resp2.targ_pdqtiles # worst % difference from a target is 0.00056% which should be good enough for most purposes


# details of what's in tp
# target_sums are the national totals implied by the geographic targets -- sum(geotargets, dims=1)
# target_calcs are the national totals we calculate from the national weights and household characteristics
# in a well-constructed problem they should be the same
# tp.target_calcs = sum(tp.wh .* tp.xmat, dims=1) by definition
