# https://github.com/SciML/Optimization.jl

# :ccsaq NLopt.LD_CCSAQ: CCSA (Conservative Convex Separable Approximations) with simple quadratic approximations (local, derivative)

function direct_nlopt(prob, result;
    maxiter=100,
    interval=1,
    whweight=.5,
    pow=4,
    kwargs...)

    # kwargs must be allowable options for NLopt that Optimization will pass through to NLopt
    kwkeys_allowed = (:stopval, ) # :show_trace, :x_tol, :g_tol,
    kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)
    println("kwargs: $kwargs_keep")

    shares0 = result.shares0

    # %% setup preallocations
    p = 1.0
    p_mshares = Array{Float64,2}(undef, prob.h, prob.s)
    p_whs = Array{Float64,2}(undef, prob.h, prob.s)
    p_calctargets = Array{Float64,2}(undef, prob.s, prob.k)
    p_pdiffs = Array{Float64,2}(undef, prob.s, prob.k)
    p_whpdiffs = Array{Float64,1}(undef, prob.h)

    println("Household weights component weight: ", whweight)

    fp = (shares, p) -> objfn_direct(shares, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled,
        p_mshares, p_whs, p_calctargets, p_pdiffs, p_whpdiffs, interval, whweight, pow)

    fpof = OptimizationFunction{true}(fp, Optimization.AutoZygote())
    fprob = OptimizationProblem(fpof, shares0, lb=zeros(length(shares0)), ub=ones(length(shares0)))  # MAIN ONE

    # NLOPT gradient-based local algorithms that can handle bounds and that do NOT use dense matrix methods
    #   I exclude slsqp because it uses dense methods
    # https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#local-gradient-based-optimization
    # so far, ccsaq is best for this application
    method = result.method
    if method==:ccsaq algorithm=:(LD_CCSAQ())
    elseif method==:lbfgs algorithm=:(LD_LBFGS())
    elseif method==:mma algorithm=:(LD_MMA())
    elseif method==:newton algorithm=:(LD_TNEWTON_PRECOND())
    elseif method==:newtonrs algorithm=:(LD_TNEWTON_PRECOND_RESTART())
    elseif method==:var1 algorithm=:(LD_VAR1())
    elseif method==:var2 algorithm=:(LD_VAR2())
    else return "ERROR: method must be one of (:ccsaq, :lbfgs, :mma, :newton, :var2)"
    end
    println("NLopt algorithm: ", algorithm)

    # opt = Optimization.solve(fprob, NLopt.eval(algorithm), maxiters=maxiter, callback=cb_direct; kwargs_keep...)
    opt = Optimization.solve(fprob, NLopt.eval(algorithm), maxiters=maxiter, callback=cb_direct; kwargs_keep...)

    # ERROR: AutoZygote does not currently support constraints
    # opt = Optimization.solve(fprob, NLopt.LD_AUGLAG(), local_method = NLopt.LD_LBFGS(), local_maxiters=10000, maxiters=maxiter)

    result.solver_result = opt
    # result.success = (opt.retcode == Symbol("true")) || (opt.minimum <= kwargs_keep[:stopval])
    #result.success = (opt.retcode == Symbol("true")) || (opt.retcode == :STOPVAL_REACHED)
    result.success = true
    result.iterations = iter_calc # opt.original.iterations
    result.shares = opt.minimizer
    return result
end


function direct_test(prob, shares0, result; maxiter=100, objscale=1.0, interval=1, whweight, kwargs...)
    # println("shares0 start: ", Statistics.quantile(vec(shares0)))
    kwkeys_allowed = (:show_trace, :x_tol, :g_tol)
    kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)

    # %% setup preallocations
    p = 1.0
    # shares0 = fill(1. / prob.s, prob.h * prob.s)
    p_mshares = Array{Float64,2}(undef, prob.h, prob.s)
    p_whs = Array{Float64,2}(undef, prob.h, prob.s)
    p_calctargets = Array{Float64,2}(undef, prob.s, prob.k)
    p_pdiffs = Array{Float64,2}(undef, prob.s, prob.k)
    p_whpdiffs = Array{Float64,1}(undef, prob.h)

    if whweight===nothing
        whweight = length(shares0) / length(p_calctargets)
    end
    println("Household weights component weight: ", whweight)

    fp = (shares, p) -> objfn_direct(shares, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled,
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
    fprob = OptimizationProblem(fpof, shares0, lb=zeros(length(shares0)), ub=ones(length(shares0)))
    # fprob = OptimizationProblem(fpof, shares0, lb=zeros(length(shares0)), ub=ones(length(shares0)), lcons=zeros(prob.h), ucons=zeros(prob.h))

    # fprob = OptimizationProblem(fpof, shares0)

    # opt = Optimization.solve(fprob, Optim.LBFGS(), maxiters=maxiter)
    # opt = Optimization.solve(fprob, Optim.GradientDescent(linesearch=LineSearches.BackTracking(order=3)), maxiters=maxiter)
    # opt = Optimization.solve(fprob, NLopt.LD_MMA(), maxiters=maxiter)
    # opt = Optimization.solve(fprob, NLopt.LD_LBFGS(), maxiters=maxiter)
    opt = Optimization.solve(fprob, NLopt.LD_CCSAQ(), maxiters=maxiter)  # Excellent
    # opt = Optimization.solve(fprob, NLopt.LD_MMA(), maxiters=maxiter)

    # ERROR: AutoZygote does not currently support constraints
    # opt = Optimization.solve(fprob, IPNewton(), maxiters=maxiter)
    # opt = Optimization.solve(fprob, Ipopt.Optimizer()) # no options other than max time; not practical

    # opt = Optimization.solve(fprob, NLopt.LD_AUGLAG(), local_method = NLopt.LD_LBFGS(), local_maxiters=10000, maxiters=maxiter)
    # opt = Optimization.solve(fprob, NLopt.LD_CCSAQ(), maxiters=maxiter)

    # LD_TNEWTON_PRECOND slow progress
    # LD_MMA pretty good
    # LD_CCSAQ ok not great
    # LD_AUGLAG
    # LD_LBFGS slow
    # LD_VAR2 about same as LD_LBFGS
    # LD_TNEWTON_PRECOND_RESTART faster than lbfgs

    result.solver_result = opt
    result.success = opt.retcode == Symbol("true")
    result.iterations = -9 # opt.original.iterations
    result.shares = opt.minimizer
    return result
end

function direct_test_scaled(prob, shares0, result; maxiter=100, interval=1, whweight=nothing, objscale=1.0, kwargs...)
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


    # opt = Optimization.solve(fprob, NLopt.LD_MMA(), maxiters=maxiter)
    # opt = Optimization.solve(fprob, NLopt.LD_LBFGS(), maxiters=maxiter)
    opt = Optimization.solve(fprob, NLopt.LD_CCSAQ(), maxiters=maxiter)  # Excellent
    # opt = Optimization.solve(fprob, NLopt.LD_CCSAQ(), maxiters=maxiter)
    # opt = Optimization.solve(fprob, NLopt.LD_TNEWTON_PRECOND_RESTART(), maxiters=maxiter)

    # ERROR: AutoZygote does not currently support constraints

    # IPNewton
    # ERROR: This optimizer requires derivative definitions for nonlinear constraints. If the problem does not have nonlinear constraints,
    #  choose a different optimizer. Otherwise define the derivative for cons using OptimizationFunction either directly or
    # automatically generate them with one of the autodiff backends
    # opt = Optimization.solve(fprob, IPNewton(), maxiters=maxiter)
    # opt = Optimization.solve(fprob, Ipopt.Optimizer()) # no options other than max time; not practical

    # opt = Optimization.solve(fprob, NLopt.LD_AUGLAG(), local_method = NLopt.LD_LBFGS(), local_maxiters=10000, maxiters=maxiter)
    # opt = Optimization.solve(fprob, NLopt.LD_CCSAQ(), maxiters=maxiter)

    # LD_TNEWTON_PRECOND slow progress
    # LD_MMA pretty good
    # LD_CCSAQ ok not great
    # LD_AUGLAG
    # LD_LBFGS slow
    # LD_VAR2 about same as LD_LBFGS
    # LD_TNEWTON_PRECOND_RESTART faster than lbfgs

    result.solver_result = opt
    result.success = opt.retcode == Symbol("true")
    result.iterations = -9 # opt.original.iterations
    result.shares = opt.minimizer ./ s_scale
    return result
end


