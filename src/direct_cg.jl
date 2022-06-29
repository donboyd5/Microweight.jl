function direct_cg(prob, result; shares0=fill(1. / prob.s, prob.h * prob.s), maxiter=100, objscale=1.0, interval=1, targweight=0.8, kwargs...)
    # caller(tp; ishares=fill(1. / tp.s, tp.h * tp.s), maxiters=10, interval=1, targweight=0.1)
    # for allowable arguments:

    kwkeys_allowed = (:show_trace, :x_tol, :g_tol)
    kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)

    # %% setup preallocations
    p = 1.0
    shares0 = fill(1. / prob.s, prob.h * prob.s)
    p_mshares = Array{Float64,2}(undef, prob.h, prob.s)
    p_whs = Array{Float64,2}(undef, prob.h, prob.s)
    p_calctargets = Array{Float64,2}(undef, prob.s, prob.k)
    p_pdiffs = Array{Float64,2}(undef, prob.s, prob.k)
    p_whpdiffs = Array{Float64,1}(undef, prob.h)

    fp = (shares, p) -> fcons(shares, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled,
        p_mshares, p_whs, p_calctargets, p_pdiffs, p_whpdiffs, interval, targweight)

    fpof = OptimizationFunction{true}(fp, Optimization.AutoZygote())
    fprob = OptimizationProblem(fpof, shares0, lb=zeros(length(shares0)), ub=ones(length(shares0)))

    # override default eta (0.4) but  use default linesearch
    opt = Optimization.solve(fprob, Optim.ConjugateGradient(linesearch = LineSearches.HagerZhang(), eta = 0.1), maxiters=maxiter)

    result.solver_result = opt
    result.success = opt.retcode == Symbol("true") || (opt.original.iterations >= 100)
    result.iterations = opt.original.iterations
    result.shares = opt.minimizer
    return result
end