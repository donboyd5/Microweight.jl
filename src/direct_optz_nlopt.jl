
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

If the chosen global optimzer employs a local optimization method a similiar set of common local
optimizer arguments exists. The common local optimizer arguments are:
    local_method (optimiser used for local optimization in global method)
    local_maxiters (the maximum number of iterations)
    local_maxtime (the maximum of time the optimization runs for)
    local_abstol (absolute tolerance in changes of the objective value)
    local_reltol (relative tolerance in changes of the objective value)
    local_options (NamedTuple of keyword arguments for local optimizer)

Some optimizer algorithms have special keyword arguments documented in the solver portion of
the documentation and their respective documentation. These arguments can be passed as kwargs... to solve.
Similiarly, the special kewyword arguments for the local_method of a global optimizer are passed as a NamedTuple to local_options.

For NLopt:
https://optimization.sciml.ai/stable/optimization_packages/nlopt/
https://nlopt.readthedocs.io/en/latest/NLopt_Reference
Beyond the common arguments the following optimizer parameters can be set as kwargs: [djb: these are not available for all algorithms]
    stopval  ...this is useful...
    xtol_rel
    xtol_abs
    constrtol_abs
    initial_step  ? https://nlopt.readthedocs.io/en/latest/NLopt_Reference/#initial-step-size
    population  ? for stochastic
    vector_storage ? https://nlopt.readthedocs.io/en/latest/NLopt_Guile_Reference/#vector-storage-for-limited-memory-quasi-newton-algorithms

# :ccsaq NLopt.LD_CCSAQ: CCSA (Conservative Convex Separable Approximations) with simple quadratic approximations (local, derivative)
=#

function direct_optz_nlopt(prob, result;
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

    # kwargs must be common options or allowable options for NLopt that Optimization will pass through to NLopt
    println("kwargs requested: ", keys(kwargs))
    kwkeys_method = (:maxtime, :abstol, :reltol)
    kwkeys_algo = (:stopval, )
    # merge the allowable sets of keys
    kwkeys_allowed = (kwkeys_method..., kwkeys_algo...)
    println("kwargs allowed: ", kwkeys_allowed)
    kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)
    println("kwargs passed on: $kwargs_keep")

    println("Household weights component weight: ", whweight)
    println("\n")

    opt = Optimization.solve(fprob, NLopt.eval(algorithm), maxiters=maxiter, callback=cb_direct; kwargs_keep...)

    # ERROR: AutoZygote does not currently support constraints
    # opt = Optimization.solve(fprob, NLopt.LD_AUGLAG(), local_method = NLopt.LD_LBFGS(), local_maxiters=10000, maxiters=maxiter)

    result.solver_result = opt
    # result.success = (opt.retcode == Symbol("true")) || (opt.minimum <= kwargs_keep[:stopval])
    # result.success = (opt.retcode == Symbol("true")) || (opt.retcode == :STOPVAL_REACHED) || (iter_calc / 2.) >= maxiter
    result.success = true
    result.iterations = iter_calc # opt.original.iterations
    result.shares = opt.minimizer
    return result
end



function OLD_direct_test_scaled(prob, shares0, result; maxiter=100, interval=1, whweight=nothing, objscale=1.0, kwargs...)
    # println("shares0 start: ", Statistics.quantile(vec(shares0)))
    kwkeys_allowed = (:show_trace, :x_tol, :g_tol)
    kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)

    println("s_scale: ", s_scale)

    shares0a = shares0 .* s_scale
    # println("shares0a: ", shares0a)

    # %% setup preallocations
    p = 1.0
    # shares0 = fill(1. / prob.s, prob.h * prob.s)
    p_mshares = Array{Float64,2}(undef, prob.h, prob.s)
    p_whs = Array{Float64,2}(undef, prob.h, prob.s)
    p_calctargets = Array{Float64,2}(undef, prob.s, prob.k)
    p_pdiffs = Array{Float64,2}(undef, prob.s, prob.k)
    p_whpdiffs = Array{Float64,1}(undef, prob.h)

    if isnothing(whweight)
        whweight = (length(shares0) / length(p_calctargets)) / (s_scale / 1.)
    end
    println("Household weights component weight: ", whweight)

    fp = (shares, p) -> objfn_direct_scaled(shares, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled,
        p_mshares, p_whs, p_calctargets, p_pdiffs, p_whpdiffs, interval, whweight)

    function cons_addup(shares, wh)
        mshares = reshape(shares, length(wh), :)
        sum(mshares, dims=2) .- 1.
    end
    cons = (shares, p) -> cons_addup(shares, prob.wh)

    fpof = OptimizationFunction{true}(fp, Optimization.AutoZygote())
    # fpof = OptimizationFunction{true}(fp, Optimization.AutoZygote(), cons=cons) # ERROR: AutoZygote does not currently support constraints
    # fpof = OptimizationFunction{true}(fp, Optimization.AutoReverseDiff(), cons=cons) # ERROR: AutoReverseDiff does not currently support constraints
    # fpof = OptimizationFunction{true}(fp, Optimization.AutoModelingToolkit(), cons=cons)
    # fpof = OptimizationFunction{true}(fp, Optimization.AutoForwardDiff(), cons=cons)

    # AutoForwardDiff
    fprob = OptimizationProblem(fpof, shares0a, lb=zeros(length(shares0)), ub=ones(length(shares0)).*s_scale)  # MAIN ONE
    # fprob = OptimizationProblem(fpof, shares0a) # FOR KRYLOV
    # fprob = OptimizationProblem(fpof, shares0, lb=zeros(length(shares0)), ub=ones(length(shares0)), lcons=zeros(prob.h), ucons=zeros(prob.h))

    # fprob = OptimizationProblem(fpof, shares0)

    # opt = Optimization.solve(fprob, Optim.LBFGS(), maxiters=maxiter)
    # opt = Optimization.solve(fprob, Optim.GradientDescent(linesearch=LineSearches.BackTracking(order=3)), maxiters=maxiter)
    # opt = Optimization.solve(fprob,
    #     Optim.ConjugateGradient(
    #         alphaguess = LineSearches.InitialConstantChange(ρ = 0.75), # best; ρ = 0.25 default
    #         linesearch = LineSearches.BackTracking(order=3), # 2nd best
    #         eta = 0.7 # best
    #         ), reltol=0.0, maxiters=maxiter)
    # opt = Optimization.solve(fprob,
    #         Optim.KrylovTrustRegion(),
    #       maxiters=maxiter, store_trace=true, show_trace=false)

    opt = Optimization.solve(fprob, NLopt.LD_CCSAQ(), maxiters=maxiter)  # Excellent

    # ERROR: AutoZygote does not currently support constraints
    result.solver_result = opt
    result.success = opt.retcode == Symbol("true")
    result.iterations = -9 # opt.original.iterations
    result.shares = opt.minimizer ./ s_scale
    return result
end


