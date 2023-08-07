
using Optimization, OptimizationNLopt, OptimizationOptimJL, Zygote

function f(x) # Rosenbrock
    obj = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    obj
end

function cb(x, obj)    
    println("obj = $obj")
    halt = obj < 0.7
    return halt
end

x0 = zeros(2)
fp = (x, p) -> f(x) # Optimization.jl syntax requires p as an argument

fpof = Optimization.OptimizationFunction{true}(fp, Optimization.AutoZygote())
fprob = Optimization.OptimizationProblem(fpof, x0)
opt = Optimization.solve(fprob, NLopt.LD_LBFGS(), callback=cb) 
opt = Optimization.solve(fprob, NLopt.LN_NELDERMEAD(), callback=cb)
opt = Optimization.solve(fprob, NLopt.LN_COBYLA(), callback=cb)


opt = Optimization.solve(fprob, Optim.LBFGS(), callback=cb) 

opt = Optimization.solve(fprob, Optim.KrylovTrustRegion(), callback=cb) 

