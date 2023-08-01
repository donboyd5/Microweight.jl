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

h = 1000  # number of households 100
k = 4 # number of characteristics each household has 4

h = 10000  # number of households 100
k = 20 # number of characteristics each household has 4

h = 100_000  # number of households 100
k = 50 # number of characteristics each household has 4

h = 300_000  # number of households 100
k = 100 # number of characteristics each household has 4

# the function mtp (make test problem) will create a random problem with these characteristics
tp = mw.mtprw(h, k, pctzero=0.3)
fieldnames(typeof(tp))

tp.xlb = 0.1
tp.xub = 10.0

# next steps here ....

mw.rwsolve(tp, approach=:minerr, method=:ccsaq)
mw.rwsolve(tp, approach=:minerr)
mw.rwsolve(tp, approach=:abc, method=:ccsaq)


mw.objfn_reweight(ones(tp.h), tp.wh, tp.xmat, tp.rwtargets, rweight=0.5)

algs = ["LD_CCSAQ", "LD_LBFGS", "LD_MMA", "LD_VAR1", "LD_VAR2", "LD_TNEWTON", "LD_TNEWTON_RESTART", "LD_TNEWTON_PRECOND_RESTART", "LD_TNEWTON_PRECOND"]
# LD_LBFGS about 0.5 seconds with rweight=0.5

iters = 10_000
rwt = 1e-4
rwt = 0.0
scaleit = false
opt1 = mw.rwsolve(tp.wh, tp.xmat, tp.rwtargets, algo=algs[1], rweight=rwt, scaling=scaleit, maxit=iters)

lb = .25
ub = 1.75

lb = .25
ub = 4.

opt1a = mw.rwsolve(tp.wh, tp.xmat, tp.rwtargets, algo=algs[1], lb=lb, ub=ub, rweight=rwt, scaling=scaleit, maxit=iters)

opt2 = mw.rwsolve(tp.wh, tp.xmat, tp.rwtargets, algo=algs[2], rweight=rwt, scaling=scaleit, maxit=iters)
opt3 = mw.rwsolve(tp.wh, tp.xmat, tp.rwtargets, algo=algs[3], rweight=rwt, scaling=scaleit, maxit=iters)
opt4 = mw.rwsolve(tp.wh, tp.xmat, tp.rwtargets, algo=algs[4], rweight=rwt, scaling=scaleit, maxit=iters)
opt5 = mw.rwsolve(tp.wh, tp.xmat, tp.rwtargets, algo=algs[5], rweight=rwt, scaling=scaleit, maxit=iters)
opt6 = mw.rwsolve(tp.wh, tp.xmat, tp.rwtargets, algo=algs[6], rweight=rwt, scaling=scaleit, maxit=iters)
opt7 = mw.rwsolve(tp.wh, tp.xmat, tp.rwtargets, algo=algs[7], rweight=rwt, scaling=scaleit, maxit=iters)
opt8 = mw.rwsolve(tp.wh, tp.xmat, tp.rwtargets, algo=algs[8], rweight=rwt, scaling=scaleit, maxit=iters)
opt9 = mw.rwsolve(tp.wh, tp.xmat, tp.rwtargets, algo=algs[9], rweight=rwt, scaling=scaleit, maxit=iters)

opt = opt1
opt = opt1a
opt = opt2
opt = opt3
opt = opt4
opt = opt5
opt = opt6
opt = opt7
opt = opt8
opt = opt9

# 6 LD_TNEWTON super fast, also 9 LD_TNEWTON_PRECOND, then 2, 8, 4, 7, 5
opt.objective
opt.solve_time
opt.u
opt.retcode

quantile(opt.u)

rwtargets_calc = tp.xmat' * (opt.u .* tp.wh)
targpdiffs = (rwtargets_calc .- tp.rwtargets) ./ tp.rwtargets 
quantile(targpdiffs)


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
pdiffs0 = tp.rwtargets_calc ./ tp.rwtargets .- 1.
quantile(pdiffs0)

## Ipopt section ----
# ???
import LinearAlgebra, OpenBLAS32_jll
LinearAlgebra.BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path)

function f(ratio)
    # part 1 get measure of difference from targets
    ratiodiffs = ratio .- 1.0
    objval = sum(ratiodiffs.^2.) / length(ratio)

  # list extra variables on the return so that they are available to the callback function
  return objval # , targdiffs, whdiffs, targstop, whstop
end

f(fill(1.2, 10))

ratio0 = fill(1., tp.h)
lower = fill(.1, tp.h)
upper = fill(10., tp.h)
lcons = tp.rwtargets * .99
ucons = tp.rwtargets * 1.01
p = 1.0

fp = (ratio, p) -> f(ratio)
fp(ratio0, p)

# cons = (res, u, p) -> res .= w'u
# function fcons(ratio, xmat, wh)
#     cons = xmat' * (ratio .* wh)
#     return cons
# end
# fcons(ratio0, tp.xmat, tp.wh)
# fcp = (ratio, p) -> fcons(ratio, xmat, wh)
# xmat = tp.xmat
# wh = tp.wh
# fcp(ratio0, p)


# cons2 = (res, x, p) -> res .= [x[1]^2 + x[2]^2]
xmat = tp.xmat
wh = tp.wh
fcons = (res, ratio, p) -> res .= xmat' * (ratio .* wh)

fpof = Optimization.OptimizationFunction{true}(fp, Optimization.AutoForwardDiff(); cons=fcons)
fprob = Optimization.OptimizationProblem(fpof, ratio0, p, lb=lower, ub=upper, lcons=lcons, ucons=ucons) # rerun this line when ratio0 changes
res = Optimization.solve(fprob, Ipopt.Optimizer())
# res2 = Optimization.solve(fprob, Optim.IPNewton()) # warning if not an interior point

lcons
xmat' * (res.u .* wh)
ucons
quantile(res.u)

## standalone ipopt
# https://chrisgeoga.com/
# https://docs.juliahub.com/StandaloneIpopt/QHju1/0.4.1/
# https://git.sr.ht/~cgeoga/StandaloneIpopt.jl
# https://git.sr.ht/~cgeoga/StandaloneIpopt.jl/tree
# https://juliahub.com/ui/Packages/StandaloneIpopt/QHju1/0.4.1
# https://discourse.julialang.org/t/ann-standaloneipopt-jl-another-option-for-using-ipopt-in-julia/86014
# https://git.sr.ht/~cgeoga/StandaloneIpopt.jl/tree/master/item/example/hs071.jl
# Sparse constraint Jacobian support via SparseDiffTools.jl. Simply pass in a sparse matrix of Bools as the jac_sparsity kwarg

using StandaloneIpopt
wh = tp.wh
xmat = tp.xmat
sparse(xmat)
!(xmat==0)
bjac = xmat .!= 0
# Bool(bjac)

bjac = sparse(xmat .!= 0)
# bjac = iszero.(xmat)

sparse(xmat )
ratio0 = ones(length(wh))
lcons = tp.rwtargets * .99999
ucons = tp.rwtargets * 1.00001

obj(ratio) = sum((ratio .- 1.0).^2.) / length(ratio)
# obj(ratio0)

fcons(store, ratio) = store .= xmat' * (ratio .* wh)

# store = zeros(tp.k)
# fcons(store, ratio0)
# A = [1 0; 2 -1]  # your numeric matrix
# B = sparse(A .> 0)

(lb, ub) = (0.1, 10.0)

constraints = Constraints(tp.k, fcons, lcons, ucons)
res = ipopt_optimize(obj, ratio0, constraints,
                    box_lower=lb, box_upper=ub)

res2 = ipopt_optimize(obj, ratio0, constraints, box_lower=lb, box_upper=ub, jac_nz=bjac)


# jac_sparsity

fieldnames(typeof(res))
res.minval
res.minimizer
opt1a.u

lcons
xmat' * (res.minimizer .* wh)
ucons

rwtargets_calc = tp.xmat' * (res.minimizer .* tp.wh)
# targdiffs = (rwtargets_calc .- rwtargets) ./ rwtargets  * 100. # allocates a tiny bit
targdiffs = (rwtargets_calc .- tp.rwtargets) # ./ 1e6 # allocates a tiny bit
pdiffs = targdiffs ./ tp.rwtargets

rwtargets_calc = tp.xmat' * (opt1a.u .* tp.wh)
# targdiffs = (rwtargets_calc .- rwtargets) ./ rwtargets  * 100. # allocates a tiny bit
targdiffs = (rwtargets_calc .- tp.rwtargets) # ./ 1e6 # allocates a tiny bit
pdiffs = targdiffs ./ tp.rwtargets


using Symbolics

fcalls = 0
function f(y,x) # in-place
  global fcalls += 1
  for i in 2:length(x)-1
    y[i] = x[i-1] - 2x[i] + x[i+1]
  end
  y[1] = -2x[1] + x[2]
  y[end] = x[end-1] - 2x[end]
  nothing
end

input = rand(30)
output = similar(input)
sparsity_pattern = Symbolics.jacobian_sparsity(f,output,input)
jac = Float64.(sparsity_pattern)

using SparseArrays
A = sparse([1, 1, 2, 3], [1, 3, 2, 3], [0, 1, 2, 0])


# function fcons(store, ratio)
#     store[1] = prod(x)
#     store[2] = sum(abs2, x)
#     store
#     end

using StandaloneIpopt

# hs071
#
# min  x1*x4*(x1+x2+x3)+x3
# s.t. x1*x2*x3*x4 >= 25
#      x1^2 + x2^2 + x3^2 + x4^2 == 40
#      1 <= x1, x2, x3, x4 <= 5

# Objective:
obj(x) = x[1]*x[4] * (x[1]+x[2]+x[3]) + x[3]

# constraints:
function eval_constraints(store, x)
store[1] = prod(x)
store[2] = sum(abs2, x)
store
end

# box values, the same for all parameters. If they weren't, you could instead
# pass in box_lower = [l_1, l_2, ..., l_n].
(b_l, b_u) = (1.0, 5.0)

# init value:
ini = [1.0, 5.0, 5.0, 1.0]

# result:
constraints = Constraints(2, eval_constraints, [25.0, 40.0], [1e22, 40.0])
res = ipopt_optimize(obj, ini, constraints,
                    box_lower=b_l, box_upper=b_u)


# cons(res, x, p) = (res .= [x[1]^2 + x[2]^2, x[1] * x[2]])
res = [0.0, 0.0]
cons(res, ratio, p) = (res .= xmat' * (ratio .* wh))
cons(res, ratio0, p)

cons(res, x, p) = (res .= [x[1]^2 + x[2]^2, x[1] * x[2]])
x1 = [1.0, 2.0]
x = x1
res = [0.0, 0.0]
res = 1.0
cons(res, x1, p)

tp.rwtargets


using Optimization, OptimizationOptimJL
rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
cons2 = (res, x, p) -> res .= [x[1]^2 + x[2]^2]
x0 = zeros(2)
p = [1.0, 100.0]
fpof = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff(); cons = cons2)
prob = Optimization.OptimizationProblem(fpof, x0, p, lcons = [-5.0], ucons = [10.0])
sol = solve(prob, IPNewton())
sol2 = solve(prob, Ipopt.Optimizer())
rosenbrock(sol.u, p)
rosenbrock(sol2.u, p)

# END Ipopt section ----



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
ratio0 = ones(tp.h)
p = 1.0
wh = tp.wh
# scale = 1e3
scale = vec(sum(abs.(tp.xmat), dims=1)) ./ size(tp.xmat)[1] 
scale = fill(1., k) # unscaled
xmat = tp.xmat ./ scale'
sum(abs.(xmat), dims=1)
mean(abs.(xmat), dims=1)

rwtargets = tp.rwtargets ./ scale
xmat' * wh


mw.objfn_reweight(ratio0, wh, xmat, rwtargets)
mw.objfn_reweight(ratio0, wh, xmat, rwtargets, scaling=false)


# set up optimization functions
fp = (ratio, p) -> f(ratio, wh, xmat, rwtargets)
fpof = Optimization.OptimizationFunction{true}(fp, Optimization.AutoZygote())
# fpof = Optimization.OptimizationFunction{true}(fp, Optimization.AutoModelingToolkit())
# fpof = Optimization.OptimizationFunction{true}(fp, Optimization.AutoForwardDiff())
# fpof = Optimization.OptimizationFunction{true}(fp, Optimization.AutoFiniteDiff())

# fprob = Optimization.OptimizationProblem(fpof, wh0, lb=zeros(length(wh0)), ub=ones(length(wh0)))
# lower = 0.1*ones(length(ratio0))
# upper = 10.0*ones(length(ratio0))
fprob = Optimization.OptimizationProblem(fpof, ratio0, lb=0.1, ub=10.0) # rerun this line when ratio0 changes
# fprob = Optimization.OptimizationProblem(fpof, res.x, lb=0.1, ub=10.0)


fp(ratio0, p)
f(ratio0, wh, xmat, rwtargets)

a = "LD_CCSAQ()"
a = :LD_CCSAQ()

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

function f(ratio, wh, xmat, rwtargets)

  # part 1 get measure of difference from targets
  wh2 = ratio .* wh
  rwtargets_calc = xmat' * wh2  
  # targdiffs = (rwtargets_calc .- rwtargets) ./ rwtargets  * 100. # allocates a tiny bit
  targdiffs = (rwtargets_calc .- rwtargets) ./ rwtargets # ./ 1e6 # allocates a tiny bit
  ss_targdiffs = mean(targdiffs.^2.0)
  objval = ss_targdiffs

# list extra variables on the return so that they are available to the callback function
return objval # , targdiffs, whdiffs, targstop, whstop
end

# f <- mw.objfn_reweight(ratio, wh, xmat, rwtargets;
# rweight=0.5,
# pow=2.0,
# targstop=true, whstop=true,
# display_progress=true)

ratio0 = ones(tp.h)
p = 1.0
wh = tp.wh
# scale = 1e3
# scale = vec(sum(abs.(tp.xmat), dims=1)) ./ size(tp.xmat)[1] 
scale = fill(1., k) # unscaled
xmat = tp.xmat ./ scale'
sum(abs.(xmat), dims=1)
mean(abs.(xmat), dims=1)

rwtargets = tp.rwtargets ./ scale
xmat' * wh


# f2 = (ratio) -> f(ratio, wh, xmat, rwtargets)
f2 = (ratio) -> mw.objfn_reweight(ratio, wh, xmat, rwtargets, rweight=0.0)
g2 = (ratio) -> ReverseDiff.gradient(f2, ratio)
# g2(g2, ratio) -> ReverseDiff.gradient!(g2, f2, ratio)

f2(ratio0)
g2(ratio0)

lower = fill(0.1, length(ratio0)) # can't use scalar
upper = fill(10., length(ratio0))

lower = fill(0.2, length(ratio0)) # can't use scalar
upper = fill(5., length(ratio0))

x = ratio0
@time res1 = spgbox(f2, (g,x) -> ReverseDiff.gradient!(g,f2,x), x, lower=lower, upper=upper, eps=1e-16, nitmax=10000, nfevalmax=20000, m=10, iprint=0)

lower = fill(0.1, length(ratio0)) # can't use scalar
upper = fill(10., length(ratio0))
f2 = (ratio) -> mw.objfn_reweight(ratio, wh, xmat, rwtargets, rweight=1e-6)
g2 = (ratio) -> ReverseDiff.gradient(f2, ratio)
@time res2 = spgbox(f2, (g,x) -> ReverseDiff.gradient!(g,f2,x), x, lower=lower, upper=upper, eps=1e-16, nitmax=5000, nfevalmax=10000, m=20, iprint=0)
# m	Integer	Number of non-monotone search steps.	10
# @time res2 = spgbox(f2, g2, x, lower=lower, upper=upper, eps=1e-10, nitmax=400, nfevalmax=2000, m=10, iprint=0)
#  65.271938 seconds (285.43 k allocations: 220.238 GiB, 8.65% gc time, 0.28% compilation time)
# fieldnames(typeof(res))

res = res1
res.f
res.nit
res.x
quantile(res.x)

opt1a.objective
opt1a.u


# (:u, :cache, :alg, :objective, :retcode, :original, :solve_time, :stats)
opt.objective # 2.974735500954226e-6
fp(ratio0, p)
fp(opt.u, p)

opt.solve_time # 12.152999877929688
opt.retcode


xyz = opt1a.u
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

cor(res.x, opt.u)


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


