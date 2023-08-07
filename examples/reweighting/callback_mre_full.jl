using Optimization, ForwardDiff, Zygote, OptimizationOptimJL, OptimizationNLopt

function rosenbrock(x, p)
    obj = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    obj
end

function cb(x, obj)    
    println("obj = $obj")
    # return true
    halt = obj < 0.25
    return halt
end


x0 = zeros(2)
_p = [1.0, 100.0]


optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, _p)
sol = solve(prob, BFGS())

sol = solve(prob, BFGS(), callback=cb)


# new start
using Optimization, OptimizationNLopt, Zygote

function f(x)
    # Rosenbrock
    obj = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    obj
end

function cb(x, obj)
    println("obj = $obj")
    return obj < 0.7
end

x0 = zeros(2)
p = ones(2)
fp = (x, p) -> f(x)  # p required as an argument per the syntax of Optimization

fpof = Optimization.OptimizationFunction{true}(fp, Optimization.AutoZygote())
fprob = Optimization.OptimizationProblem(fpof, x0) #, lb=lb, ub=ub) 
opt = Optimization.solve(fprob, NLopt.LD_LBFGS(), callback=cb) 




opt = Optimization.solve(fprob, NLopt.LD_LBFGS()) 








function f(x)
    obj = sum((x .- 1).^2)
    obj
end


opt = Optimization.solve(fprob, NLopt.LD_LBFGS()) 
opt = Optimization.solve(fprob, NLopt.LD_LBFGS(), callback=cb) 

# opt = Optimization.solve(fprob, NLopt.LD_CCSAQ()) 

opt2 = Optimization.solve(fprob, NLopt.LD_CCSAQ(), callback=cb) 


opt = Optimization.solve(fprob, NLopt.LD_LBFGS()) 



n = 2
x0 = zeros(n)
p = ones(n)
lb = fill(-10., n)
ub = fill(9., n)
fp = (x, p) -> f(x)

fp(x0, p)
fp(lb, p)
fp(ub, p)
fp(p, p)

function cb(x, obj)    
    println("obj = $obj")
    # return true
    # halt = obj < 0.7
    obj < 0.7
    # return halt
end

function cb(x, p, obj)
    println("obj = $obj")
    return true
    # halt = obj < 0.7
    obj < 0.7
    # return halt
end
# fpof = Optimization.OptimizationFunction{true}(fp, Optimization.AutoForwardDiff())