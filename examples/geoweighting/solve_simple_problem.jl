import Microweight as mw
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
h = 100  # number of households
s = 8  # number of states, regions, etc.
k = 4 # number of characteristics each household has
# the function mtp (make test problem) will create a random problem with these characteristics
tp = mw.mtp(h, s, k)


## explore what's in tp
fieldnames(typeof(tp))
# (:wh, :xmat, :geotargets, :h, :k, :s, :target_sums, :target_calcs, :target_diffs, :wh_scaled, :xmat_scaled, :geotargets_scaled)

# wh, xmat, and geotargets are fields you will always need to solve for state weights
# we'll look at other fields later - they're not needed to solve for state weights
tp.wh # h x 1 matrix
tp.xmat # h x k matrix
tp.geotargets # s x k matrix


## solve the problem using defaults
# this will use poisson approach and lm_lsqfit
# this uses the Levenberg Marquardt algorithm, a widely used algorithm to solve least-squares problems
# this should work well for most problems - you may not care to investigate other approaches and methods unless
resp_lm_lsqfit = mw.geosolve(tp)
# geosolve prints results of each iteration; we'll discuss they're meaning elsewhere

## explore what's in resp1
fieldnames(typeof(resp_lm_lsqfit))
# (:approach, :method, :success, :iterations, :eseconds, :sspd, :beta, :beta0, :shares, :shares0, :whs, :wh_calc, :wh_pdiffs, :wh_pdqtiles, :geotargets_calc, :targ_pdiffs, :targ_pdqtiles, :solver_result, :problem)
resp_lm_lsqfit.approach # :poisson
resp_lm_lsqfit.method # :lm_lsqfit
resp_lm_lsqfit.sspd  # sum of squared percentage differences from the geotargets - a summary measure of how well we did
resp_lm_lsqfit.eseconds # elapsed seconds (including any needed compilation time)
resp_lm_lsqfit.whs  # weights for households, for each state -- the key result

resp_lm_lsqfit.geotargets_calc # s x k matrix of geographic targets calculated using the state weights
# we can compare this to tp.geotargets - our intended target values - we want them to be extremely close
resp_lm_lsqfit.targ_pdiffs # percentage differences of calculated targets from intended targets
# i.e., this is (resp_lm_lsqfit.geotargets_calc ./ tp.geotargets .- 1.0) * 100.0
# we'd like all of these to be zero
resp_lm_lsqfit.targ_pdqtiles # quantiles of target percentage differences, hopefully all near zero

# there are similar results for the calculated state weights - we want each household's sum of state weights to be near its national weight
resp_lm_lsqfit.wh_pdiffs # percentage differences of household weights summed across states from given national weights
# calculated as (resp_lsqfit.wh_calc - tp.wh) ./ tp.wh * 100.
resp_lm_lsqfit.wh_pdqtiles # quantiles of percentage differences of sums of state weights, relative to given national state weights - should be near zero

resp_lm_lsqfit.solver_result # info returned by the solver, which will vary by solver

## solve by different methods, still using poisson approach
# why multiple methods? sometimes some algorithms work better than others
# use another poisson method - lm_lsoptim, and state the poisson approach explicity (not necessary, as it is default)
resp_lm_lsoptim = mw.geosolve(tp, approach=:poisson, method=:lm_lsoptim)

# methods from the minpack package
resp_lm_minpack = mw.geosolve(tp, approach=:poisson, method=:lm_minpack) # another implementation of Levenburg Marquardt
resp_hybr_minpack = mw.geosolve(tp, approach=:poisson, method=:hybr_minpack) # uses Powell's dogleg method

# methods from the NLOPT.jl package
resp_ccsaq = mw.geosolve(tp, approach=:poisson, method=:ccsaq)
resp_mma = mw.geosolve(tp, approach=:poisson, method=:mma)
# not progressing: newton, newtonrs, lbfgs_nlopt, var1, var2 -- Something is wrong because lbfgs from optim works

# methods from the nlsolve package
resp_newton_nlsolve = mw.geosolve(tp, approach=:poisson, method=:newton_nlsolve)
resp_trust_nlsolve = mw.geosolve(tp, approach=:poisson, method=:trust_nlsolve)

# methods from the Optim package
# optim_methods = (:cg, :gd, :lbfgs_optim, :krylov)
# resp = mw.geosolve(tp, approach=:poisson, method=:krylov)
resp_cg = mw.geosolve(tp, approach=:poisson, method=:cg)
resp_gd = mw.geosolve(tp, approach=:poisson, method=:gd)
resp_krylov = mw.geosolve(tp, approach=:poisson, method=:krylov)
resp_lbfgs_optim = mw.geosolve(tp, approach=:poisson, method=:lbfgs_optim)

# methods from the optimisers package
# optimisers_methods = (:adam,  :descent, :momentum, :nesterov)
resp_adam = mw.geosolve(tp, approach=:poisson, method=:adam, pow=2, maxiter=3000)
resp_descent = mw.geosolve(tp, approach=:poisson, method=:descent) # can't improve on starting point
resp_momentum = mw.geosolve(tp, approach=:poisson, method=:momentum) # can't improve on starting point
resp_nesterov = mw.geosolve(tp, approach=:poisson, method=:nesterov) # can't improve on starting point

## direct approach
# nlopt_methods = (:ccsaq, :lbfgs_nlopt, :mma, :newton, :newtonrs, :var1, :var2)
# optim_methods = (:cg, :gd, :lbfgs_optim)
# optimisers_methods = (:adam, :nesterov, :descent, :momentum)
# nlopt
resd_ccsaq = mw.geosolve(tp, approach=:direct, maxiter=10_000)
resd_lbfgs_nlopt = mw.geosolve(tp, approach=:direct, method=:lbfgs_nlopt)
resd_newton = mw.geosolve(tp, approach=:direct, method=:newton)
resd_var2 = mw.geosolve(tp, approach=:direct, method=:var2)
Statistics.quantile(vec(resd_ccsaq.whs))
resd_ccsaq.wh_pdqtiles
resd_ccsaq.targ_pdqtiles
resd_ccsaq.eseconds

# optim
resd_cg = mw.geosolve(tp, approach=:direct, method=:cg)
resd_gd = mw.geosolve(tp, approach=:direct, method=:gd)
resd_lbfgs_optim = mw.geosolve(tp, approach=:direct, method=:lbfgs_optim)

# optimisers -- look for negative whs!
# (:adam,  :descent, :momentum, :nesterov)
resd_adam = mw.geosolve(tp, approach=:direct, method=:adam, pow=2, maxiter=20000)
resd_descent = mw.geosolve(tp, approach=:direct, method=:descent) # can't improve on start
resd_momentum = mw.geosolve(tp, approach=:direct, method=:momentum) # can't improve on start
resd_nesterov = mw.geosolve(tp, approach=:direct, method=:nesterov)
Statistics.quantile(vec(resd_adam.whs))

resd_krylov = mw.geosolve(tp, approach=:direct, method=:krylov, pow=2) # not much progress on real problems
Statistics.quantile(vec(resd_krylov.whs))

# resd_xxx = mw.geosolve(tp, approach=:direct, method=:xxx)
# resd_xxx = mw.geosolve(tp, approach=:direct, method=:xxx)
# resd_xxx = mw.geosolve(tp, approach=:direct, method=:xxx)
# resd_xxx = mw.geosolve(tp, approach=:direct, method=:xxx)


cor(vec(resp_lm_lsqfit.whs), vec(resd_ccsaq.whs))

# tp = mw.get_taxprob(8)

## compare results
# did all three methods have very low sums of squared percentage differences from targets?
# yes - compare sspd for several
# differences across methods reflect different options and stopping criteria and do not necessarily indicate
# that one was better than the other

# the real question is whether results are "good enough"
# for example resp_lsoptim has a much higher sspd than resp_lsqfit:
resp_lsqfit.sspd
resp_lsoptim.sspd
# but the sum of squared percentage differences from targets was about 3 millionths of a percentage point
# that may be good enough for most purposes

# but , but are the state weight-sums close enough to national weights, and are
# calculated targets close enough to intended targets?

resp_lm_minpack.sspd
resp_hybr_minpack.sspd
xx.sspd
xx.sspd
xx.sspd
xx.sspd
xx.sspd
xx.sspd



resp2.wh_pdqtiles # state-weight sums for each household are close to the household's national weight (all % diffs are near 0)
resp_adam.targ_pdqtiles # worst % difference from a target is 0.00056% which should be good enough for most purposes


# details of what's in tp
# target_sums are the national totals implied by the geographic targets -- sum(geotargets, dims=1)
# target_calcs are the national totals we calculate from the national weights and household characteristics
# in a well-constructed problem they should be the same
# tp.target_calcs = sum(tp.wh .* tp.xmat, dims=1) by definition
