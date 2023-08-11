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

h = 500_000  # number of households 100
k = 200 # number of characteristics each household has 4

# the function mtp (make test problem) will create a random problem with these characteristics
tp = mw.mtprw(h, k, pctzero=0.3);

h=100; k=4; tp = mw.mtprw(h, k, pctzero=0.3);
h=1_000; k=6; tp = mw.mtprw(h, k, pctzero=0.3);
h=10_000; k=20; tp = mw.mtprw(h, k, pctzero=0.3);
h=100_000; k=50; tp = mw.mtprw(h, k, pctzero=0.3);
h=300_000; k=100; tp = mw.mtprw(h, k, pctzero=0.3);

h=100_000; k=27; tp = mw.mtprw(h, k, pctzero=0.7);

fieldnames(typeof(tp))

function qpdiffs(ratio)
  rwtargets_calc = tp.xmat' * (ratio .* tp.wh)
  targpdiffs = (rwtargets_calc .- tp.rwtargets) ./ tp.rwtargets 
  Statistics.quantile(targpdiffs)
end

# LBFGS seems to be best when ratio error is most important, CCSAQ when target error is most important
algs = ["LD_CCSAQ", "LD_LBFGS", "LD_MMA", "LD_VAR1", "LD_VAR2", "LD_TNEWTON", "LD_TNEWTON_RESTART", "LD_TNEWTON_PRECOND_RESTART", "LD_TNEWTON_PRECOND"]


##############################################################################
##
## Compare results under alternative methods
##
##############################################################################

res1 = mw.rwsolve(tp, approach=:minerr, method="spg", lb=.1, ub=10.0, rweight=1e-6, maxiters=2000, print_interval=10, targstop=0.01);
res2 = mw.rwsolve(tp, approach=:minerr, method="LD_CCSAQ", lb=.1, ub=10.0, rweight=1e-6, maxiters=2000, print_interval=10, targstop=0.01); # returns ones
res3 = mw.rwsolve(tp, approach=:minerr, method="LD_LBFGS", lb=.1, ub=10.0, rweight=1e-6, maxiters=2000, print_interval=10, targstop=0.01);
res4 = mw.rwsolve(tp, approach=:minerr, method="LD_MMA", lb=.1, ub=10.0, rweight=1e-6, maxiters=2000, print_interval=10, targstop=0.01);
res5 = mw.rwsolve(tp, approach=:constrain, method="ipopt", lb=.1, ub=10.0, constol=.01)
res6 = mw.rwsolve(tp, approach=:constrain, method="tulip", lb=.1, ub=10.0, constol=.01)

# res 7 seems to hang up -- maybe scaling?
res7= mw.rwsolve(tp, approach=:minerr, method="LBFGS", lb=.1, ub=10.0, rweight=1e-6, maxiters=2000, print_interval=10, targstop=0.01); # does not seem to work well

# m = hcat(res1.x, res2.x, res3.x, res4.x)
m = hcat(res1.x, res2.x, res3.x, res4.x, res5.x, res6.x)
m
cor(m)

qpdiffs(res1.x)
qpdiffs(res2.x)
qpdiffs(res3.x)
qpdiffs(res4.x)
qpdiffs(res5.x)
qpdiffs(res6.x)

quantile(res1.x)
quantile(res2.x)
quantile(res3.x)
quantile(res4.x)
quantile(res5.x)
quantile(res6.x)

##############################################################################
##
## Misc playing around
##
##############################################################################


# run through different methods and with different parameters
res= mw.rwsolve(tp, approach=:minerr, print_interval=1);
res= mw.rwsolve(tp, approach=:minerr, print_interval=1, targstop=.0351);

res= mw.rwsolve(tp, approach=:minerr, method="LD_LBFGS", print_interval=1, targstop=.01);

res= mw.rwsolve(tp, approach=:minerr);
res= mw.rwsolve(tp, approach=:minerr, method="LD_LBFGS", print_interval=10);
res= mw.rwsolve(tp, approach=:minerr, method="LD_CCSAQ", print_interval=10);
res= mw.rwsolve(tp, approach=:minerr, method="LD_LBFGS", lb=.2, ub=2.0, print_interval=10);
res= mw.rwsolve(tp, approach=:minerr, method="LD_LBFGS", lb=.2, ub=2.0, maxiters=2000, print_interval=10);
res= mw.rwsolve(tp, approach=:minerr, method="LD_LBFGS", lb=.1, ub=10.0, rweight=0.0001, maxiters=2000, print_interval=100);
res= mw.rwsolve(tp, approach=:minerr, method="LD_LBFGS", lb=.1, ub=10.0, rweight=0.0001, maxiters=2000, print_interval=10, targstop=0.005);
res= mw.rwsolve(tp, approach=:minerr, method=algs[3], lb=.1, ub=10.0, rweight=0.0001, maxiters=2000, print_interval=100);
res= mw.rwsolve(tp, approach=:minerr, method=algs[8]);

fieldnames(typeof(res))

res= mw.rwsolve(tp, approach=:minerr, method="LBFGS", print_interval=1);
res= mw.rwsolve(tp, approach=:minerr, method="LBFGS", print_interval=1, lb=.1, ub=10.0, rweight=0.);

res= mw.rwsolve(tp, approach=:minerr, method="LBFGS", print_interval=100, lb=.1, ub=5.0, rweight=1e-6, targstop=.01);

res= mw.rwsolve(tp, approach=:minerr, method="LBFGS", lb=.1, ub=10.0, rweight=0.0001, maxiters=2000, print_interval=100);


# res= mw.rwsolve(tp, approach=:minerr, method="KrylovTrustRegion", print_interval=10);


res.solve_time
res.objective
quantile(res.u)

qpdiffs(ones(tp.h))
qpdiffs(res.u)

res2 = mw.rwsolve(tp, approach=:minerr, method="spg", lb=.1, ub=10.0)
res2 = mw.rwsolve(tp, approach=:minerr, method="spg", lb=.1, ub=10.0, targstop=.3109)
res2 = mw.rwsolve(tp, approach=:minerr, method="spg", lb=.1, ub=10.0, rweight=0.0001, targstop=.012)

res2= mw.rwsolve(tp, approach=:minerr, method="spg", lb=.1, ub=10.0, rweight=0.0, maxiters=2000, print_interval=10, targstop=0.01);

res2 = mw.rwsolve(tp, approach=:minerr, method="spg", lb=.1, ub=10.0, rweight=1e-9, targstop=.01)

tp2 = tp
rnums = .1 .* randn(k) .+ 1.0
b = tp.rwtargets_calc .* rnums
tp2.rwtargets = b

res2 = mw.rwsolve(tp2, approach=:minerr, method="spg", lb=.1, ub=10.0, rweight=1e-9, targstop=.01)
res2 = mw.rwsolve(tp2, approach=:minerr, method="spg", lb=.0, ub=2.0, rweight=1e-9, targstop=.01)

res2 = mw.rwsolve(tp, approach=:minerr, method="spg", lb=.1, ub=10.0, rweight=0.0)
res2 = mw.rwsolve(tp, approach=:minerr, method="spg", lb=.1, ub=10.0, rweight=1e-5)
res2 = mw.rwsolve(tp, approach=:minerr, method="spg", lb=.5, ub=1.5, rweight=0.0)
fieldnames(typeof(res2))
res2.f

kres = res2
qpdiffs(ones(tp.h))
quantile(kres.x)
qpdiffs(kres.x)

mw.rwsolve(tp, approach=:minerr, method="xyz")

res3 = mw.rwsolve(tp, approach=:constrain)

res3 = mw.rwsolve(tp, approach=:constrain, lb=.1, ub=10.0, constol=.01)
res3 = mw.rwsolve(tp, approach=:constrain, lb=.1, ub=5.0, constol=.01)

results = fieldnames(typeof(res3))

res3.objval
quantile(res3.x)
qpdiffs(res3.x)


mw.rwsolve(tp, approach=:something)

res3 = mw.rwsolve(tp, approach=:constrain)

res3 = mw.rwsolve(tp, approach=:constrain, method="tulip")
quantile(res3.x)
qpdiffs(res3.x)

##############################################################################
##
## Tulip my original, with variants
##
##############################################################################

using JuMP, Tulip

h=10; k=2; tp = mw.mtprw(h, k, pctzero=0.1);
# h=100; k=4; tp = mw.mtprw(h, k, pctzero=0.1);

A = tp.xmat .* tp.wh
A1 = transpose(A)
# A1 = A'
A2 = -A1
b = tp.rwtargets_calc

N = size(A1)[2]
scale = (N / 1000.) ./ sum(abs.(A1), dims=2)
scale = 1.0

A1s = scale .* A1
A2s = scale .* A2
bs = scale .* b

tol = .50
model = Model(Tulip.Optimizer)
set_optimizer_attribute(model, "OutputLevel", 1)  # 0=disable output (default), 1=show iterations
set_optimizer_attribute(model, "IPM_IterationsLimit", 100) 

@variable(model, r[1:N] >= 0)
@variable(model, s[1:N] >= 0)

@objective(model, Min, sum(r[i] + s[i] for i in 1:N))

# bound on top by tolerance
@constraint(model, [i in 1:N], r[i] + s[i] <= tol)

# Ax = b
@constraint(model, [i in 1:length(b)], sum(A1[i,j] * r[j] + A2[i,j] * s[j] for j in 1:N) == b[i])


optimize!(model)

##############################################################################
##
## Tulip - this is the one to use
##
##############################################################################
# issue: https://github.com/PSLmodels/taxdata/issues/381
# chusloj's original solver.jl before ANY changes I made can be found here:
# https://github.com/PSLmodels/taxdata/blob/e014837b98f83258bfc425a89ba79c368890f801/puf_stage2/stage2.py
# https://github.com/PSLmodels/taxdata/blob/e014837b98f83258bfc425a89ba79c368890f801/puf_stage2/dataprep.py
# https://github.com/PSLmodels/taxdata/blob/e014837b98f83258bfc425a89ba79c368890f801/cps_stage2/solver.jl

# r_val = array['r']
# s_val = array['s']
# z_val = (1. + r_val - s_val) * s006 * 100

using JuMP, Tulip
using Random, Distributions
using HiGHS

function print_constraints(m::Model)
  for con_ref in all_constraints(m, VariableRef, MOI.EqualTo{Float64})
      println(con_ref)
  end
  for con_ref in all_constraints(m, VariableRef, MOI.LessThan{Float64})
      println(con_ref)
  end
  # Add more loops for other types of constraints if needed
end


h=10; k=2; tp = mw.mtprw(h, k, pctzero=0.1);
h=100; k=10; tp = mw.mtprw(h, k, pctzero=0.1);
h=1_000; k=20; tp = mw.mtprw(h, k, pctzero=0.1);
h=10_000; k=50; tp = mw.mtprw(h, k, pctzero=0.3);
h=100_000; k=100; tp = mw.mtprw(h, k, pctzero=0.3);
h=100_000; k=27; tp = mw.mtprw(h, k, pctzero=0.7);
# h=100; k=4; tp = mw.mtprw(h, k, pctzero=0.1);

A = tp.xmat .* tp.wh
rnums = .05 .* randn(k) .+ 1.0
b = tp.rwtargets_calc .* rnums
# b = tp.rwtargets

ctol = .01
lower = b .- ctol * abs.(b)
upper = b .+ ctol * abs.(b)

N = size(A)[1]
scale = vec((N / 1000.) ./ sum(abs.(A), dims=1))
# scale = 1.0

As = A .* scale'
bs = b .* scale
lowers = lower .* scale
uppers = upper .* scale


# tol = 10.0

##############################################################################
##
## the taxdata approach
##
##############################################################################
model = Model(Tulip.Optimizer)
set_optimizer_attribute(model, "OutputLevel", 1)  # 0=disable output (default), 1=show iterations
set_optimizer_attribute(model, "IPM_IterationsLimit", 100)  # default 100 seems to be enough
# set_optimizer_attribute(model, "Threads", 1)  
# set_optimizer_attribute(model, "Threads", 24)  

# model = Model(HiGHS.Optimizer)
# set_attribute(model, "presolve", "on")
# set_attribute(model, "time_limit", 60.0)
# set_attribute(model, "threads", 12)
# set_optimizer_attribute(model, "solver", "ipm")  
# set_optimizer_attribute(model, "solver", "simplex")  


# for both solvers....
tol = 10.0
@variable(model, 0.0 <= r[1:N] <= tol) # djb r is the amount above 1 e.g., 1 + 0.40, goes with A1s
@variable(model, 0.0 <= s[1:N] <= tol) # djb s is the amount below 1 e.g., 1 - 0.40, goes with A2s

# @variable(model, 0.0 <= r[1:N])
# @variable(model, 0.0 <= s[1:N]) 
# @variable(model,  0.0 <= r[1:N] <= tol / 2.)
# @variable(model,  0.0 <= s[1:N] <= tol / 2.)

# @variable(model,  0.0 <= r[1:N], start=0.5)
# @variable(model,  0.0 <= s[1:N], start=0.5)

# @variable(model,  r[1:N])
# @variable(model,  s[1:N])
# @objective(model, Min, sum(r[j] + s[j] for j in 1:N));  # djb would be clearer to use j as the index here

# @objective(model, Min, sum(r .+ s)); 

@objective(model, Min, sum(r + s)); 

# Ax = b  - use the scaled matrices and vector; equality constraints
initval = vec(sum(As, dims=1))

# compare these two!!!
# @constraint(model, initval .+ (As' * r) .- (As' * s) .== bs); # note .==
# @constraint(model, initval + (As' * r) - (As' * s) .== bs); # note .==; broadcasting not needed on LHS with equal size vectors


@constraint(model, lowers .<= initval + (As' * r) - (As' * s) .<= uppers); #
# @constraint(model, [i in 1:length(b)], sum(A1[i,j] * r[j] + A2[i,j] * s[j] for j in 1:N) == b[i])


@constraint(model, 0.0 .<= (1.0 .+ r .- s) .<= tol); # all dots needed for broadcasting
# @constraint(model, (1.0 .+ r .- s) .>= 0.0);  # must keepf broadcasting?
# @constraint(model, (1.0 .+ r .- s) .<= tol);  # must keepf broadcasting?

# @constraint(model, (1.0 + r - s) .>= 0.0);  # get rid of broadcasting?
# @constraint(model, (1.0 + r - s) .<= tol);  

# print_constraints(model)
optimize!(model);

# termination_status(model)
objective_value(model)
simplex_iterations(model)
barrier_iterations(model)

# Did we satisfy constraints?
r_vec = value.(r)
s_vec = value.(s)

# bs
# initval .+ (As' * r_vec) .- (As' * s_vec)
# initval + (As' * r_vec) - (As' * s_vec) 

# x = 1.0 .+ r_vec .- s_vec  # note the .+
x = 1.0 .+ (r_vec - s_vec) # note the .+ this works too - must broadcast when adding 1.0
quantile(x)
b_calc = A' * x
lower
upper
b
check = vec(b_calc) ./ b

q = (0, .1, .25, .5, .75, .9, 1)
quantile!(check, q)

# Are the ratios of new weights to old weights in bounds (within tolerances)?
quantile!(x, q)



##############################################################################
##
## Exploration
##
##############################################################################
using Optim

function f(x)
  return x[1]^2 + x[2]^2
end

result = Optim.KrylovTrustRegion(f, [1.0, 1.0], 1e-6, bounds = [[0, 10], [0, 10]])
result = Optim.KrylovTrustRegion(f, [1.0, 1.0], 1e-6, [[0, 10], [0, 10]])

println(result)
using Optim

# Define your function and its gradient
function f(x)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function g!(G, x)
    G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    G[2] = 200.0 * (x[2] - x[1]^2)
end

# Initial guess
initial_x = [0.0, 0.0]

# Optimize using KrylovTrustRegion
res = optimize(f, g!, initial_x, KrylovTrustRegion())

# Display the result
println(res.minimizer)
println(res.minimum)

# Initial guess
initial_x = [0.0, 0.0]

# Set bounds
lower_bounds = [-1.0, -1.0]
upper_bounds = [2.0, 2.0]

# Set up the Fminbox with KrylovTrustRegion as the inner optimizer
inner_optimizer = Optim.KrylovTrustRegion()
res = optimize(f, g!, lower_bounds, upper_bounds, initial_x, Fminbox(inner_optimizer))

res = optimize(f, g!, initial_x, Fminbox(inner_optimizer))


##############################################################################
##
## Test equivalency of expressions for JuMP Tulip
##
##############################################################################
# @constraint(model, initval .+ (A' * r) .- (A' * s) == b);        
# @constraint(model, [i in 1:length(b)], sum(A1[i,j] * r[j] + A2[i,j] * s[j] for j in 1:N) == b[i])

# sum(r[i] + s[i] for i in 1:N)
# [i in 1:N], r[i] + s[i] <= tol
# [i in 1:length(b)], sum(A1[i,j] * r[j] + A2[i,j] * s[j] for j in 1:N) == b[i]

[i in 1:length(b)], sum(A1[i,j] * r[j] + A2[i,j] * s[j] for j in 1:N) == b[i]

h=10
k=2
Random.seed!(3); rnums = .05 .* randn(k) .+ 1.0; r = randn(h); s = randn(k)

tp = mw.mtprw(h, k)
A = tp.xmat .* tp.wh
b = tp.rwtargets_calc

A' * r

m, n = size(A)
sum([[A[i, j]*r[i] for j in 1:n] for i in 1:m])

using JuMP, GLPK

n = 3
# model = Model(GLPK.Optimizer)
model = Model(Tulip.Optimizer)

@variable(model, x[1:n])
@variable(model, y[1:n])
@constraint(model, con[i=1:n], x[i] + y[i] == 10)

x_values = [3, 4, 5]  # Example values
y_values = [7, 6, 5]  # These values are chosen so that x[i] + y[i] = 10 for all i

x_values = [3., 4., 5.]  # Example values
y_values = [7., 6., 5.]  

for i in 1:n
    set_value(x[i], x_values[i])
    set_value(y[i], y_values[i])
end

model = Model()
@variable(model, x)
@constraint(model, con, x <= 5)

x_value = 3
set_value(con, x_value) # 3

point = Dict(x => 1.9);
primal_feasibility_report(model, point)

@variable(model, 0.0 <= r[1:n])

# chuslog approach
b = [-18.0, -45.0] # correct
b = [-18.0, -5.0] # incorrect -- but chusloj still says true
b = [-8.0, -5.0] # incorrect

b = [-18.0, -5.0] 
A1 = [1.0 2.0 3.0;
      4.  5. 6.]
A2 = -A1
N = 3
r = [1., 2., 3.]; s = [4., 5., 6.]
i = 2
[i in 1:length(b)], sum(A1[i,j] * r[j] + A2[i,j] * s[j] for j in 1:N) == b[i]

# Dummy values
N = 3
b = [1.0, 2.0, 3.0]
A1 = rand(3,3)
A2 = rand(3,3)
r = rand(3)
s = rand(3)

result1 = [sum(A1[i,j] * r[j] + A2[i,j] * s[j] for j in 1:N) for i in 1:length(b)]
result2 = A1 * r + A2 * s


# my approach
A = A1'
(A' * r) .- (A' * s) == b # one element only true if all are true
(A' * r) .- (A' * s) .== b  # two element bit vector

# Rescale variables in the problem and their associated coefficients to make the magnitudes of all coefficients in the 1e-4 to 1e4 range. 
# For example, that might mean rescaling a variable from measuring distance in centimeters to kilometers