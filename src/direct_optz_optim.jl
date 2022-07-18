
#= DOCUMENTATION
https://github.com/SciML/Optimization.jl
https://optimization.sciml.ai/stable/
https://optimization.sciml.ai/dev/

The arguments to solve are common across all of the optimizers. These common arguments are:
    maxiters (the maximum number of iterations)
    maxtime (the maximum of time the optimization runs for)
    abstol (absolute tolerance in changes of the objective value)
    reltol (relative tolerance in changes of the objective value)
    callback (a callback function)


The following special keyword arguments which are not covered by the common solve arguments can be used with Optim.jl optimizers:
https://optimization.sciml.ai/stable/optimization_packages/optim/
    x_tol: Absolute tolerance in changes of the input vector x, in infinity norm. Defaults to 0.0.
    g_tol: Absolute tolerance in the gradient, in infinity norm. Defaults to 1e-8. For gradient free methods, this will control the main convergence tolerance, which is solver specific.
    f_calls_limit: A soft upper limit on the number of objective calls. Defaults to 0 (unlimited).
    g_calls_limit: A soft upper limit on the number of gradient calls. Defaults to 0 (unlimited).
    h_calls_limit: A soft upper limit on the number of Hessian calls. Defaults to 0 (unlimited).
    allow_f_increases: Allow steps that increase the objective value. Defaults to false.
        Note that, when setting this to true, the last iterate will be returned as the minimizer even if the objective increased.
    store_trace: Should a trace of the optimization algorithm's state be stored? Defaults to false.
    show_trace: Should a trace of the optimization algorithm's state be shown on stdout? Defaults to false.
    extended_trace: Save additional information. Solver dependent. Defaults to false.
    trace_simplex: Include the full simplex in the trace for NelderMead. Defaults to false.
    show_every: Trace output is printed every show_everyth iteration.
(:x_tol, :g_tol, :f_calls_limit, :g_calls_limit, :h_calls_limit, :allow_f_increases, :store_trace, :show_trace, :extended_trace, :show_every)

# https://github.com/JuliaNLSolvers/Optim.jl/blob/master/src/multivariate/optimize/interface.jl

No to NGMRES and OACCEL - I haven't figured out yet how to make them work - check docs again

opt = Optimization.solve(fprob,
    Optim.NewtonTrustRegion(), maxiters=maxiter, callback=cb_direct)  # newt does not work without figuring box constraints
nlprecon = Optim.GradientDescent(alphaguess=LineSearches.InitialStatic(alpha=1e-4,scaled=true),
        linesearch=LineSearches.Static())
Default size of subspace that OACCEL accelerates over is `wmax = 10`
oacc10 = Optim.OACCEL(nlprecon=nlprecon, wmax=10)
opt = Optimization.solve(fprob,
 Optim.OACCEL(nlprecon=nlprecon, wmax=10), maxiters=maxiter, callback=cb_direct)
optimize(UP.objective(prob), UP.gradient(prob), prob.initial_x, oacc10)

=#

function direct_optz_optim(prob, result;
    maxiter,
    whweight,
    pow,
    targstop, whstop,
    kwargs...)

    # %% setup preallocations
    p = 1.0
    p_mshares = Array{Float64,2}(undef, prob.h, prob.s)
    p_whs = Array{Float64,2}(undef, prob.h, prob.s)
    p_calctargets = Array{Float64,2}(undef, prob.s, prob.k)
    p_pdiffs = Array{Float64,2}(undef, prob.s, prob.k)
    p_whpdiffs = Array{Float64,1}(undef, prob.h)

    fp = (shares, p) -> objfn_direct(shares, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled,
        p_mshares, p_whs, p_calctargets, p_pdiffs, p_whpdiffs, whweight, pow, targstop, whstop)

    fpof = OptimizationFunction{true}(fp, Optimization.AutoZygote())
    fprob = OptimizationProblem(fpof, result.shares0, lb=zeros(length(result.shares0)), ub=ones(length(result.shares0)))

    method = result.method
    if method==:cg algorithm=:(ConjugateGradient())
    elseif method==:gd algorithm=:(GradientDescent())
    elseif method==:lbfgs_optim algorithm=:(LBFGS())
    # I do not allow krylov because it cannot use box constraints - Fminbox does not allow it
    # but consider penalty
    else return "ERROR: method must be one of (:cg, gd, :lbfgs_optim)"
    end
    println("Optim algorithm: ", algorithm)

    # println("kwargs requested: ", keys(kwargs))
    kwkeys_method = (:maxtime, :abstol, :reltol)
    kwkeys_algo = (:x_tol, :g_tol, :f_calls_limit, :g_calls_limit, :h_calls_limit, :allow_f_increases, :store_trace, :show_trace, :extended_trace, :show_every)
    kwargs_defaults = Dict() # :stopval => 1e-4
    kwargs_use = kwargs_keep(kwargs; kwkeys_method=kwkeys_method, kwkeys_algo=kwkeys_algo, kwargs_defaults=kwargs_defaults)

    println("Household weights component weight: ", whweight)
    println("\n")

    opt = Optimization.solve(fprob,
        Optim.eval(algorithm), maxiters=maxiter, callback=cb_direct; kwargs_use...)

    result.solver_result = opt
    # result.success = opt.retcode == Symbol("true")
    result.success = true
    result.iterations = opt.original.iterations
    result.shares = opt.minimizer
    return result
end

# function direct_cg_good(prob, result; shares0=fill(1. / prob.s, prob.h * prob.s), maxiter=100, objscale=1.0, interval=1, whweight, kwargs...)
#     # caller(tp; ishares=fill(1. / tp.s, tp.h * tp.s), maxiters=10, interval=1, targweight=0.1)
#     # for allowable arguments:

#     kwkeys_allowed = (:show_trace, :x_tol, :g_tol)
#     kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)

#     # %% setup preallocations
#     p = 1.0
#     shares0 = fill(1. / prob.s, prob.h * prob.s)
#     p_mshares = Array{Float64,2}(undef, prob.h, prob.s)
#     p_whs = Array{Float64,2}(undef, prob.h, prob.s)
#     p_calctargets = Array{Float64,2}(undef, prob.s, prob.k)
#     p_pdiffs = Array{Float64,2}(undef, prob.s, prob.k)
#     p_whpdiffs = Array{Float64,1}(undef, prob.h)

#     if whweight===nothing
#         whweight = length(shares0) / length(p_calctargets)
#     end
#     println("Household weights component weight: ", whweight)

#     fp = (shares, p) -> objfn_direct(shares, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled,
#         p_mshares, p_whs, p_calctargets, p_pdiffs, p_whpdiffs, interval, whweight)

#     fpof = OptimizationFunction{true}(fp, Optimization.AutoZygote())
#     # fpof = OptimizationFunction{true}(fp, Optimization.AutoFiniteDiff()) # AutoFiniteDiff AutoReverseDiff(compile=false)
#     # fpof = OptimizationFunction(fp, Optimization.AutoModelingToolkit(), shares0, p,
#     #     grad = true, hess = true, sparse = true,
#     #     checkbounds = true,  linenumbers = true)

#     # fpof = OptimizationFunction(fp, Optimization.AutoModelingToolkit(true, true))
#     # fprob = OptimizationProblem(fpof, shares0, p, lb=zeros(length(shares0)), ub=ones(length(shares0)))

#     fprob = OptimizationProblem(fpof, shares0, lb=zeros(length(shares0)), ub=ones(length(shares0)))
#     # fprob = OptimizationProblem(fpof, shares0, lb=ones(length(shares0))*1e-3, ub=ones(length(shares0))*.999)

#     # override default eta (0.4) but  use default linesearch
#     # opt = Optimization.solve(fprob,
#     # Optim.ConjugateGradient(
#     #     # alphaguess https://github.com/JuliaNLSolvers/Optim.jl/blob/master/docs/src/algo/linesearch.md
#     #     # LineSearches also allows the user to decide how the initial step length for the line search algorithm is chosen.
#     #     # This is set with the alphaguess keyword argument for the Optim algorithm. The default procedure varies.
#     #     # alphaguess = LineSearches.InitialStatic(), # default; best
#     #     # alphaguess = LineSearches.InitialPrevious(), # NO
#     #     # alphaguess = LineSearches.InitialQuadratic(), # NO
#     #     #alphaguess = LineSearches.InitialConstantChange(ρ = 0.25), # fast, default
#     #     # alphaguess = LineSearches.InitialConstantChange(ρ = 0.75), # best; ρ = 0.25 default
#     #     # alphaguess = LineSearches.InitialHagerZhang(),
#     #     # alphaguess = LineSearches.InitialHagerZhang(α0=1.0),
#     #     # │   delta: Float64 0.1
#     #     # │   sigma: Float64 0.9
#     #     # │   alphamax: Float64 Inf
#     #     # │   rho: Float64 5.0
#     #     # │   epsilon: Float64 1.0e-6
#     #     # │   gamma: Float64 0.66
#     #     # │   linesearchmax: Int64 50
#     #     # │   psi3: Float64 0.1
#     #     # │   display: Int64 0
#     #     # │   mayterminate: Base.RefValue{Bool}
#     #     # │ , Flat()), NaN, 0.001, Optim.var"#49#51"())
#     #     # linesearch = LineSearches.HagerZhang(), # best
#     #     linesearch = LineSearches.HagerZhang(rho=5.0, psi3=0.1), # defaults seem best
#     #     # linesearch = LineSearches.BackTracking(order=3), # 2nd best
#     #     eta = 0.4 # best
#     #     ), reltol=1e-24, maxiters=maxiter)

#     # best for stub 9
#     opt = Optimization.solve(fprob,
#         Optim.ConjugateGradient(
#             # alphaguess https://github.com/JuliaNLSolvers/Optim.jl/blob/master/docs/src/algo/linesearch.md
#             # LineSearches also allows the user to decide how the initial step length for the line search algorithm is chosen.
#             # This is set with the alphaguess keyword argument for the Optim algorithm. The default procedure varies.
#             # alphaguess = LineSearches.InitialStatic(), # default; best
#             # alphaguess = LineSearches.InitialPrevious(), # NO
#             # alphaguess = LineSearches.InitialQuadratic(), # NO
#             # alphaguess = LineSearches.InitialConstantChange(ρ = 0.25), # fast, default
#             alphaguess = LineSearches.InitialConstantChange(ρ = 0.75), # best; ρ = 0.25 default
#             # alphaguess = LineSearches.InitialHagerZhang(),
#             # alphaguess = LineSearches.InitialHagerZhang(α0=1.0),
#             # │   delta: Float64 0.1
#             # │   sigma: Float64 0.9
#             # │   alphamax: Float64 Inf
#             # │   rho: Float64 5.0
#             # │   epsilon: Float64 1.0e-6
#             # │   gamma: Float64 0.66
#             # │   linesearchmax: Int64 50
#             # │   psi3: Float64 0.1
#             # │   display: Int64 0
#             # │   mayterminate: Base.RefValue{Bool}
#             # │ , Flat()), NaN, 0.001, Optim.var"#49#51"())
#             # linesearch = LineSearches.HagerZhang(), # best
#             # linesearch = LineSearches.HagerZhang(rho=5.0, psi3=0.1), # defaults seem best
#             linesearch = LineSearches.BackTracking(order=3), # 2nd best
#             eta = 0.7 # best
#             ), reltol=0.0, maxiters=maxiter)

#     result.solver_result = opt
#     result.success = opt.retcode == Symbol("true")
#     result.iterations = opt.original.iterations
#     result.shares = opt.minimizer
#     return result
# end