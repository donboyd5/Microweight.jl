#=
Notes
Why won't lbgfs go past 1 iteration??
https://stats.stackexchange.com/questions/126251/how-do-i-force-the-l-bfgs-b-to-not-stop-early-projected-gradient-is-zero


https://github.com/SciML/Optimization.jl
https://optimization.sciml.ai/stable/
https://optimization.sciml.ai/stable/optimization_packages/optim/#optim
maybe https://mtk.sciml.ai/dev/systems/OptimizationSystem/
https://optimization.sciml.ai/stable/API/optimization_problem/ for OptimizationProblem
https://optimization.sciml.ai/stable/API/optimization_function/ OptimizationFunction

using OptimizationOptimJL -- Optim.jl solvers
using OptimizationNLopt -- NLOPT solvers

-- OptimizationProblem and OptimizationFunction ----------------------------------------------------------------
https://optimization.sciml.ai/stable/API/optimization_problem/
https://optimization.sciml.ai/stable/API/optimization_function/

full names are SciMLBase.OptimizationFunction etc.

OptimizationProblem{iip}(f, x, p = SciMLBase.NullParameters(),;
                        lb = nothing,
                        ub = nothing,
                        lcons = nothing,
                        ucons = nothing,
                        sense = nothing,
                        kwargs...)

iip means isinplace
If f is a standard Julia function, it is automatically converted into an OptimizationFunction with NoAD(), i.e.,
  no automatic generation of the derivative functions.

---- SciMLBase.OptimizationFunction
------- Automatic differentiation
OptimizationFunction(f,AutoZygote()) will use Zygote.jl to define all of the necessary functions. Note that if
 any functions are defined directly, the auto-AD definition does not overwrite the user's choice.

The choices for the auto-AD fill-ins with quick descriptions are:
  AutoForwardDiff(): The fastest choice for small optimizations
  AutoReverseDiff(compile=false): A fast choice for large scalar optimizations
  AutoTracker(): Like ReverseDiff but GPU-compatible
  AutoZygote(): The fastest choice for non-mutating array-based (BLAS) functions
  AutoFiniteDiff(): Finite differencing, not optimal but always applicable
  AutoModelingToolkit(): The fastest choice for large scalar optimizations

---- solve
https://optimization.sciml.ai/stable/API/solve/
solve(prob::OptimizationProblem, alg::AbstractOptimizationAlgorithm; kwargs...)

Keyword Arguments
The arguments to solve are common across all of the optimizers. These common arguments are:
  maxiters (the maximum number of iterations)
  maxtime (the maximum of time the optimization runs for)
  abstol (absolute tolerance in changes of the objective value)
  reltol (relative tolerance in changes of the objective value)
  callback (a callback function)


## Optim solvers of interest --------------------------------
Optim.ConjugateGradient()
Optim.GradientDescent()
Optim.LBFGS()
Optim.NewtonTrustRegion()
Optim.Newton()
Optim.KrylovTrustRegion()

-- Optim kwargs --------------------------------
https://optimization.sciml.ai/stable/optimization_packages/optim/#optim

-- common to all solvers --------------------------------
x_tol: Absolute tolerance in changes of the input vector x, in infinity norm. Defaults to 0.0.
g_tol: Absolute tolerance in the gradient, in infinity norm. Defaults to 1e-8. For gradient free methods, this will control the main convergence tolerance, which is solver specific.
f_calls_limit: A soft upper limit on the number of objective calls. Defaults to 0 (unlimited).
g_calls_limit: A soft upper limit on the number of gradient calls. Defaults to 0 (unlimited).
h_calls_limit: A soft upper limit on the number of Hessian calls. Defaults to 0 (unlimited).
allow_f_increases: Allow steps that increase the objective value. Defaults to false. Note that, when setting this to true, the last iterate will be returned as the minimizer even if the objective increased.
store_trace: Should a trace of the optimization algorithm's state be stored? Defaults to false.
show_trace: Should a trace of the optimization algorithm's state be shown on stdout? Defaults to false.
extended_trace: Save additional information. Solver dependent. Defaults to false.
trace_simplex: Include the full simplex in the trace for NelderMead. Defaults to false.
show_every: Trace output is printed every show_everyth iteration.

-- Optim.KrylovTrustRegion(): A Newton-Krylov method with Trust Regions
initial_delta: The starting trust region radius
delta_hat: The largest allowable trust region radius
eta: When rho is at least eta, accept the step.
rho_lower: When rho is less than rho_lower, shrink the trust region.
rho_upper: When rho is greater than rhoupper, grow the trust region (though no greater than deltahat).
Defaults:
initial_delta = 1.0
delta_hat = 100.0
eta = 0.1
rho_lower = 0.25
rho_upper = 0.75

optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff();cons= cons)
prob = Optimization.OptimizationProblem(optprob, x0, p)
sol = solve(prob, Optim.KrylovTrustRegion())

-- Examples: ----------------------------------------------------------------

f = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())

prob = OptimizationProblem(f, x0, p)
sol = solve(prob, Optim.BFGS())
sol = solve(prob, Optim.LBFGS())

 prob = OptimizationProblem(f, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
 sol = solve(prob, Fminbox(GradientDescent()))

 sol = solve(prob, IPNewton())

 sol = solve(prob, Optim.KrylovTrustRegion())

prob = OptimizationProblem(optprob, x0)
sol = solve(prob, OptimizationNLopt.Opt(:LN_BOBYQA, 2))

sol = solve(prob, OptimizationNLopt.Opt(:LD_LBFGS, 2))

prob = OptimizationProblem(optprob, x0, lb=[-1.0, -1.0], ub=[0.8, 0.8])
sol = solve(prob, Opt(:LD_LBFGS, 2))

sol = solve(prob, Opt(:G_MLSL_LDS, 2), nstart=2, local_method = Opt(:LD_LBFGS, 2), maxiters=10000)

# LineSearches
https://github.com/JuliaNLSolvers/LineSearches.jl
Available line search algorithms
In the docs we show how to choose between the line search algorithms in Optim.

HagerZhang (Taken from the Conjugate Gradient implementation by Hager and Zhang, 2006)
MoreThuente (From the algorithm in More and Thuente, 1994)
BackTracking (Described in Nocedal and Wright, 2006)
StrongWolfe (Nocedal and Wright)
Static (Takes the proposed initial step length.)
Available initial step length procedures
The package provides some procedures to calculate the initial step length that is passed to the line search algorithm. See the docs for its usage in Optim.

InitialPrevious (Use the step length from the previous optimization iteration)
InitialStatic (Use the same initial step length each time)
InitialHagerZhang (Taken from Hager and Zhang, 2006)
InitialQuadratic (Propose initial step length based on a quadratic interpolation)
InitialConstantChange (Propose initial step length assuming constant change in step length)

KrylovTrustRegion(; initial_radius::Real = 1.0,
                    max_radius::Real = 100.0,
                    eta::Real = 0.1,
                    rho_lower::Real = 0.25,
                    rho_upper::Real = 0.75,
                    cg_tol::Real = 0.01) =
                    KrylovTrustRegion(initial_radius, max_radius, eta,
                                  rho_lower, rho_upper, cg_tol)

=#

#region
# - abc
# - def
#endregion


function algo_optz(prob, beta0, result; maxiter=100, objscale, kwargs...)
    # for allowable arguments:

    kwkeys_allowed = (:show_trace, :x_tol, :g_tol)
    kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)

    fbeta = (beta, p) -> objfn(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled) .* objscale
    f = OptimizationFunction(fbeta, Optimization.AutoZygote())
    fprob = OptimizationProblem(f, beta0)
    # opt = Optimization.solve(fprob,
    #   Optim.LBFGS(; alphaguess = LineSearches.InitialStatic(), linesearch = LineSearches.BackTracking()),
    #   maxiters=2000, g_tol=1e-12, show_trace=true, show_every=100)

    opt = Optimization.solve(fprob,
      Optim.KrylovTrustRegion(; initial_radius = 0.1, max_radius = 100.0,
       eta = 0.1, rho_lower=0.101, rho_upper=0.75,
       cg_tol=0.01),
      maxiters=500, show_trace=true, show_every=10)

    result.solver_result = opt
    result.success = opt.retcode == Symbol("true") || (opt.original.iterations >= 100)
    result.iterations = opt.original.iterations
    result.beta = opt.minimizer

    return result
end