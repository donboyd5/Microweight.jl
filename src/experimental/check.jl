

f2(beta) = sum(beta.^4 .- beta.^2)

beta0 = [0., 1., 2., 3.]
f2(beta0)

using Optimization

f = OptimizationFunction(f2, Optimization.AutoForwardDiff())
fprob = OptimizationProblem(f, beta0)

opt = solve(fprob, Optim.BFGS())

rosenbrock(x,p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0,100.0]
rosenbrock(x0, p)
f = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(f, x0, p)
sol = solve(prob, Optim.BFGS())

rose1(x, p=nothing) =  (- x[1])^2 + (x[2] - x[1]^2)^2
f = OptimizationFunction(rose1, Optimization.AutoForwardDiff())
prob = OptimizationProblem(f, x0, p)
sol = solve(prob, Optim.BFGS())


y = 7.
z = 3.
x0 = [1., 2., 3.]
f3(x0, y, z)
f3(x, y, z) = sum(x.^4 .- x.^2 .+ y .+ z)
f3c = (x, p) -> f3(x, y, z)
f = OptimizationFunction(f3c, Optimization.AutoForwardDiff())
prob = OptimizationProblem(f, x0)
sol = solve(prob, Optim.BFGS(), show_trace=true)
sol.minimizer
f3c(sol.minimizer, nothing)
f3(sol.minimizer, y, z)




