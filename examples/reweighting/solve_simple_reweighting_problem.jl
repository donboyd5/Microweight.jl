using Revise
import Microweight as mw  # Revise doesn't work for changes to type definitions
using Statistics
using LineSearches

using Optimization
using NLopt
using Optim
using OptimizationMOI, Ipopt
using ModelingToolkit
using Optimisers

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

h = 1000  # number of households 100
k = 4 # number of characteristics each household has 4

h = 10000  # number of households 100
k = 20 # number of characteristics each household has 4

h = 100_000  # number of households 100
k = 50 # number of characteristics each household has 4

h = 300_000  # number of households 100
k = 100 # number of characteristics each household has 4

# the function mtp (make test problem) will create a random problem with these characteristics
tp = mw.mtprw(h, k, pctzero=0.2)
fieldnames(typeof(tp))

tp.h
tp.k
tp.wh
tp.xmat

# TEMPORARY: for h 10 k 2 create a feasible problem
# tp.rwtargets
# tp.rwtargets[1]
# tp.rwtargets[1] = 22900.
tp.rwtargets

tp.rwtargets_calc
tp.rwtargets_diff
tp.rwtargets_calc ./ tp.rwtargets .- 1.



# opt = Optimization.solve(fprob, NLopt.eval(algorithm), maxiters=maxiter, callback=cb_direct; kwargs_use...)

function f(ratio, wh, xmat, rwtargets)

    # part 1 get measure of difference from targets
    wh2 = ratio .* wh
    rwtargets_calc = xmat' * wh2  
    # targdiffs = (rwtargets_calc .- rwtargets) ./ rwtargets  * 100. # allocates a tiny bit
    targdiffs = (rwtargets_calc .- rwtargets) ./ rwtargets # ./ 1e6 # allocates a tiny bit
    ss_targdiffs = sum(targdiffs.^2.0)
    objval = ss_targdiffs

  # list extra variables on the return so that they are available to the callback function
  return objval # , targdiffs, whdiffs, targstop, whstop
end


# scale = (h / 1000.) ./ sum(abs.(xmat), dims=1)

# prep and scaling  
ratio0 = ones(length(tp.wh))
p = 1.0
wh = tp.wh
# scale = 1e3
scale = vec(sum(abs.(tp.xmat), dims=1)) ./ size(tp.xmat)[1] 
# scale = [1.0, 1.0]'  # unscaled
xmat = tp.xmat ./ scale'
sum(abs.(xmat), dims=1)
mean(abs.(xmat), dims=1)

rwtargets = tp.rwtargets ./ scale
xmat' * wh


# set up optimization functions
fp = (ratio, p) -> f(ratio, wh, xmat, rwtargets)
fpof = Optimization.OptimizationFunction{true}(fp, Optimization.AutoZygote())
# fpof = Optimization.OptimizationFunction{true}(fp, Optimization.AutoModelingToolkit())
# fpof = Optimization.OptimizationFunction{true}(fp, Optimization.AutoForwardDiff())
# fpof = Optimization.OptimizationFunction{true}(fp, Optimization.AutoFiniteDiff())

# fprob = Optimization.OptimizationProblem(fpof, wh0, lb=zeros(length(wh0)), ub=ones(length(wh0)))
# lower = 0.1*ones(length(ratio0))
# upper = 10.0*ones(length(ratio0))
fprob = Optimization.OptimizationProblem(fpof, ratio0, lb=0.1, ub=10.0)


fp(ratio0, p)
f(ratio0, wh, xmat, rwtargets)


# NLOPT optimizers
algorithm=:(LD_CCSAQ()) # best on real-world problems
algorithm=:(LD_LBFGS())  # best
algorithm=:(LD_LBFGS(M=20))  
algorithm=:(LD_MMA()) # also best
algorithm=:(NLopt.LD_VAR1())
algorithm=:(NLopt.LD_VAR2())
algorithm=:(NLopt.LD_TNEWTON()) # fast, not so accurate
algorithm=:(NLopt.NLopt.LD_TNEWTON_RESTART()) #  accurate
algorithm=:(NLopt.NLopt.LD_TNEWTON_PRECOND_RESTART())  # fast, not so accurate
algorithm=:(NLopt.LD_TNEWTON_PRECOND()) # failure
opt = Optimization.solve(fprob, NLopt.eval(algorithm), maxiters=1000, reltol=1e-16)

# Optim optimizers
# ConjugateGradient() GradientDescent() LBFGS()
algorithm=:(ConjugateGradient()) # little advancement
algorithm=:(GradientDescent())
algorithm=:(LBFGS())
algorithm=:(LBFGS(; m=2))
# algorithm=:(LBFGS(; m=20))
# algorithm=:(BFGS()) takes too much memory
algorithm=:(KrylovTrustRegion()) # need to specify a valid inner optimizer for Fminbox

opt = Optimization.solve(fprob, Optim.eval(algorithm), maxiters=100) # , show_trace=true, show_every=100) 

# other optimizers
# algorithm=:(Nesterov(0.0001, 0.9)) # can't use bounds
# algorithm=:(Descent()) # can't use bounds
# algorithm=:(Momentum(.001, .7)) # .01 .9 # can't use bounds
# algorithm=:(Adam(0.0001, (.9, .999))) # 0.5, 0.25 better, .1 much better # can't use bounds
# opt = Optimization.solve(fprob, Optimisers.eval(algorithm), maxiters=10)

# SPGBox
# https://m3g.github.io/SPGBox.jl/stable
using SPGBox
using ReverseDiff
f2 = (ratio) -> f(ratio, wh, xmat, rwtargets)
lower = 0.1*ones(length(ratio0))
upper = 10*ones(length(ratio0))
x = ratio0
@time res = spgbox(f2, (g,x) -> ReverseDiff.gradient!(g,f2,x), x, lower=lower, upper=upper, eps=1e-10, nitmax=500, nfevalmax=2000, m=10, iprint=1)
#  65.271938 seconds (285.43 k allocations: 220.238 GiB, 8.65% gc time, 0.28% compilation time)
# fieldnames(typeof(res))
res.f
res.nit
res.x



# (:u, :cache, :alg, :objective, :retcode, :original, :solve_time, :stats)
opt.objective # 0.26740274997483204
fp(ratio0, p)
fp(opt.u, p)

opt.solve_time
opt.retcode


xyz = opt.u
xyz = res.x
tp.rwtargets
rwtargets_calc = tp.xmat' * (xyz .* tp.wh)
# targdiffs = (rwtargets_calc .- rwtargets) ./ rwtargets  * 100. # allocates a tiny bit
targdiffs = (rwtargets_calc .- tp.rwtargets) # ./ 1e6 # allocates a tiny bit
pdiffs = targdiffs ./ tp.rwtargets
sum(pdiffs.^2.0)


q = (0, .1, .25, .5, .75, .9, 0.95, 0.99, 1.0)
quantile!(pdiffs, q)
quantile!(abs.(pdiffs), q)

pdiffs0 = tp.xmat' * tp.wh ./ tp.rwtargets .- 1.0
quantile!(pdiffs0, q)
quantile!(abs.(pdiffs0), q)

f(ratio0, wh, xmat, rwtargets)

ss_targdiffs = sum(targdiffs.^2.0)


algo=:(ConjugateGradient()) # also best but can hang
algo=:(GradientDescent()) # can hang
algo=:(LBFGS()) # also best
algo=:(NelderMead()) # also best
# algo=:(IPNewton())
# algo=:(Newton()) 
# algo=:(NGMRES())
# algo=:(KrylovTrustRegion())
# fpof2 = Optimization.OptimizationFunction{true}(fp, Optimization.AutoZygote()), lcons=0.1*ones(length(ratio0)), ucons=10*ones(length(ratio0))
# fprob2 = Optimization.OptimizationProblem(fpof2, ratio0)
opt2 = Optimization.solve(fprob, Optim.eval(algo), maxiters=100)
opt2.objective
opt2.solve_time


tp.rwtargets
rwtargets_calc = tp.xmat' * (opt2.u .* tp.wh)
# targdiffs = (rwtargets_calc .- rwtargets) ./ rwtargets  * 100. # allocates a tiny bit
targdiffs = (rwtargets_calc .- tp.rwtargets) # ./ 1e6 # allocates a tiny bit
ss_targdiffs = sum(targdiffs.^2.0)




opt3 = Optimization.solve(fprob, Ipopt.Optimizer(), maxiters=1000)






# scaling: determine a scaling vector with one value per constraint
#  - the goal is to keep coefficients reasonably near 1.0
#  - multiply each row of A1 and A2 by its specific scaling constant
#  - multiply each element of the b target vector by its scaling constant
#  - current approach: choose scale factors so that the sum of absolute values in each row of
#    A1 and of A2 will equal the total number of records / 1000; maybe we can improve on this

	# scale = (N / 1000.) ./ sum(abs.(A1), dims=2)

    #     A1s = scale .* A1
	# A2s = scale .* A2
	# bs = scale .* b

    # using Optim
    # function f(x)
    #     temp=(1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    #     println("f=",temp)
    #     return temp
    # end
    # result = Optim.optimize(f, [1.0 1.0], LBFGS(), Optim.Options(show_trace=true,extended_trace=true,iterations=1))



function f2(ratio)
    # part 1 get measure of difference from targets
    wh2 = ratio .* wh
    rwtargets_calc = xmat' * wh2  
    # targdiffs = (rwtargets_calc .- rwtargets) ./ rwtargets  * 100. # allocates a tiny bit
    targdiffs = (rwtargets_calc .- rwtargets) ./ rwtargets # ./ 1e6 # allocates a tiny bit
    ss_targdiffs = sum(targdiffs.^2.0)
    objval = ss_targdiffs

    # list extra variables on the return so that they are available to the callback function
    return objval # , targdiffs, whdiffs, targstop, whstop
end
    

ratio0 = ones(length(tp.wh))
wh = tp.wh
# scale = 1e3
scale = sum(abs.(tp.xmat), dims=1)
scale = [1.0, 1.0]'
xmat = tp.xmat ./ scale
tp.rwtargets
rwtargets = tp.rwtargets ./ scale
xmat' * wh
p = 1.0

# res = Optim.optimize(f2, ratio0, LBFGS(), Optim.Options(show_trace=true,extended_trace=true,iterations=10))

# res = Optim.optimize(f2, ratio0, LBFGS(); autodiff = :forward)

lower = 0.1*ones(length(ratio0))
upper = 10*ones(length(ratio0))

# inner_optimizer = GradientDescent()
inner_optimizer = LBFGS(linesearch=LineSearches.BackTracking(order=3, iterations=10))
res2 = Optim.optimize(f2, lower, upper, ratio0, Fminbox(inner_optimizer), Optim.Options(iterations = 50, outer_iterations = 50); autodiff = :forward)


# lower = [1.25, -2.1]
# upper = [Inf, Inf]
# initial_x = [2.0, 2.0]
# inner_optimizer = GradientDescent()
# res2 = Optim.optimize(f2, lower, upper, ratio0, Fminbox(inner_optimizer); autodiff = :forward)
# fieldnames(typeof(res2))
res2.minimum
res2.iterations
res2.minimizer

tp.wh
tp.xmat
tp.rwtargets
tp.xmat' * tp.wh
rwtargets_calc = tp.xmat' * (res2.minimizer .* tp.wh)
# targdiffs = (rwtargets_calc .- rwtargets) ./ rwtargets  * 100. # allocates a tiny bit
targdiffs = (rwtargets_calc .- tp.rwtargets) # ./ 1e6 # allocates a tiny bit
pdiffs = targdiffs ./ tp.rwtargets

q = (0, .1, .25, .5, .75, .9, 1)
quantile!(pdiffs, q)

tmp = tp.xmat .* (res2.minimizer .* tp.wh)

# https://julianlsolvers.github.io/Optim.jl/stable/#user/minimization/



l = 0.1
u = 10.0
d4 = OnceDifferentiable(f2)
res = Optim.optimize(f2, ratio0, l, u, Fminbox()) 


# fieldnames(typeof(res))
res.minimum
res.iterations
res.minimizer


function f(x::Vector, grad::Vector)
    if length(grad) > 0
        ...set grad to gradient, in-place...
    end
    return ...value of f(x)...
end




tmp = mw.objfn_reweight(
    tp.wh, tp.xmat,
    tp.rwtargets,
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
# kwkeys_method = (:maxtime, :abstol, :reltol)
# kwkeys_algo = (:stopval, )
# kwargs_defaults = Dict(:stopval => 1e-4)
# kwargs_use = kwargs_keep(kwargs; kwkeys_method=kwkeys_method, kwkeys_algo=kwkeys_algo, kwargs_defaults=kwargs_defaults)

println("Household weights component weight: ", whweight)

#see direct_optz_nlopt.jl
# fp = (shares, p) -> objfn_direct(shares, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled,
# p_mshares, p_whs, p_calctargets, p_pdiffs, p_whpdiffs, whweight, pow, targstop, whstop)

fp = (wh, p) -> mw.objfn_reweight(wh, tp.xmat, tp.rwtargets,
    whweight=0.5,
    pow=2.0,
    targstop=true, whstop=true,
    display_progress=true)

# https://docs.sciml.ai/Optimization/stable/API/optimization_function/
# AutoForwardDiff(): The fastest choice for small optimizations
# AutoReverseDiff(compile=false): A fast choice for large scalar optimizations
# AutoTracker(): Like ReverseDiff but GPU-compatible
# AutoZygote(): The fastest choice for non-mutating array-based (BLAS) functions
# AutoFiniteDiff(): Finite differencing, not optimal but always applicable
# AutoModelingToolkit(): The fastest choice for large scalar optimizations
fpof = Optimization.OptimizationFunction{true}(fp, Optimization.AutoModelingToolkit())

fpof = Optimization.OptimizationFunction{true}(fp, Optimization.AutoZygote())

# fpof = Optimization.OptimizationFunction{true}(fp, Optimization.AutoForwardDiff())
# AutoFiniteDiff()
# fpof = Optimization.OptimizationFunction{true}(fp, Optimization.AutoFiniteDiff())

wh0 = tp.wh
# fprob = Optimization.OptimizationProblem(fpof, wh0, lb=zeros(length(wh0)), ub=ones(length(wh0)))
fprob = Optimization.OptimizationProblem(fpof, wh0, lb=.1 .* wh0, ub=10. .* wh0)
algorithm=:(LD_CCSAQ())
algorithm=:(LD_LBFGS())
algorithm=:(LD_MMA())

p = 1.0
fp(wh0, p)
opt = Optimization.solve(fprob, NLopt.eval(algorithm), maxiters=100000, stopval=1e-8)


# fieldnames(typeof(opt))
opt.objective
# https://docs.sciml.ai/SciMLBase/stable/interfaces/Solutions/#retcodes
# ReturnCode.MaxIters
opt.retcode
opt.stats
opt.solve_time

targs =  tp.xmat' * opt.u
tp.rwtargets

targdiffs = (targs .- tp.rwtargets) ./ tp.rwtargets  * 1. # allocates a tiny bit
ss_targdiffs = sum(targdiffs.^2.)


tp.rwtargets_calc
fieldnames(typeof(tp))

# obj check
targs =  tp.xmat' * opt.u


