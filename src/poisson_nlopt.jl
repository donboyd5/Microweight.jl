# https://github.com/SciML/Optimization.jl

# :ccsaq NLopt.LD_CCSAQ: CCSA (Conservative Convex Separable Approximations) with simple quadratic approximations (local, derivative)

function poisson_nlopt(prob, result;
    maxiter,
    pow,
    targstop,
    whstop,
    objscale,
    kwargs...)

    # kwargs must be allowable options for NLopt that Optimization will pass through to NLopt
    kwkeys_allowed = (:stopval, ) # :show_trace, :x_tol, :g_tol,
    kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)
    println("kwargs: $kwargs_keep")

    fp = (beta, p) -> objfn_poisson(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled, pow, targstop, whstop, objscale)

    fpof = OptimizationFunction{true}(fp, Optimization.AutoZygote())

    beta0 = result.beta0
    fprob = OptimizationProblem(fpof, beta0)
    # fprob = OptimizationProblem(fpof, shares0, lb=zeros(length(shares0)), ub=ones(length(shares0)))  # MAIN ONE

    # NLOPT gradient-based local algorithms that can handle bounds and that do NOT use dense matrix methods
    #   I exclude slsqp because it uses dense methods
    # https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#local-gradient-based-optimization
    # so far, ccsaq is best for this application
    method = result.method
    if method==:ccsaq algorithm=:(LD_CCSAQ())
    elseif method==:lbfgs_nlopt algorithm=:(LD_LBFGS())
    elseif method==:mma algorithm=:(LD_MMA())
    elseif method==:newton algorithm=:(LD_TNEWTON_PRECOND())
    elseif method==:newtonrs algorithm=:(LD_TNEWTON_PRECOND_RESTART())
    elseif method==:var1 algorithm=:(LD_VAR1())
    elseif method==:var2 algorithm=:(LD_VAR2())
    else return "ERROR: method must be one of (:ccsaq, :lbfgs, :mma, :newton, :var2)"
    end
    println("NLopt algorithm: ", algorithm)

    opt = Optimization.solve(fprob, NLopt.eval(algorithm), maxiters=maxiter, callback=cb_poisson; kwargs_keep...)
    # opt = Optimization.solve(fprob, NLopt.eval(algorithm), maxiters=maxiter; kwargs_keep...)

    result.solver_result = opt
    # result.success = (opt.retcode == Symbol("true")) || (opt.minimum <= kwargs_keep[:stopval])
    # result.success = (opt.retcode == Symbol("true")) || (opt.retcode == :STOPVAL_REACHED) || (iter_calc / 2.) >= maxiter
    result.success = true
    result.iterations = iter_calc # opt.original.iterations
    result.beta = opt.minimizer
    return result
end
