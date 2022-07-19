
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

function direct_optz_optimisers(prob, result;
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

    fp = (shares, p) -> objfn_direct_negpen(shares, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled,
        p_mshares, p_whs, p_calctargets, p_pdiffs, p_whpdiffs, whweight, pow, targstop, whstop)

    fpof = OptimizationFunction{true}(fp, Optimization.AutoZygote())
    fprob = OptimizationProblem(fpof, result.shares0, lb=zeros(length(result.shares0)), ub=ones(length(result.shares0)))

    method = result.method
    if method==:nesterov algorithm=:(Nesterov(0.0001, 0.9))
    #   elseif method==:lbfgs_optim algorithm=:(LBFGS(; m=100))
    elseif method==:descent algorithm=:(Descent())
    elseif method==:momentum algorithm=:(Momentum(.001, .7)) # .01 .9
    # elseif method==:adam algorithm=:(Adam(0.05)) # 0.5, 0.25 better, .1 much better
    elseif method==:adam algorithm=:(Adam(0.0001, (.9, .999))) # 0.5, 0.25 better, .1 much better
    else return "ERROR: method must be one of (:nesterov, )"
    end
    println("Optim algorithm: ", algorithm)

    kwkeys_method = (:maxtime, :abstol, :reltol)
    kwkeys_algo = (:x_tol, :g_tol, :f_calls_limit, :g_calls_limit, :h_calls_limit, :allow_f_increases, :store_trace, :show_trace, :extended_trace, :show_every)
    kwargs_defaults = Dict() # :stopval => 1e-4
    kwargs_use = kwargs_keep(kwargs; kwkeys_method=kwkeys_method, kwkeys_algo=kwkeys_algo, kwargs_defaults=kwargs_defaults)

    println("Household weights component weight: ", whweight)
    println("\n")

    opt = Optimization.solve(fprob,
        Optimisers.eval(algorithm), maxiters=maxiter, callback=cb_direct; kwargs_use...)

    result.solver_result = opt
    # result.success = opt.retcode == Symbol("true")
    result.success = true
    result.iterations = -999 #opt.original.iterations
    result.shares = opt.minimizer
    return result
end
