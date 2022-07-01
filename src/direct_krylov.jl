function direct_krylov(prob, shares0, result; maxiter=100, objscale=1.0, interval=1, whweight, kwargs...)

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

    fp = (shares, p) -> objfn_direct_negpen(shares, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled,
        p_mshares, p_whs, p_calctargets, p_pdiffs, p_whpdiffs, interval, whweight)

    fpof = OptimizationFunction{true}(fp, Optimization.AutoZygote())
    # fprob = OptimizationProblem(fpof, shares0, lb=zeros(length(shares0)), ub=ones(length(shares0)))
    fprob = OptimizationProblem(fpof, shares0)

    # opt = Optimization.solve(fprob,
    # Optim.KrylovTrustRegion(),
    # maxiters=maxiter, store_trace=true, show_trace=false)
    # https://github.com/SciML/Optimization.jl/blob/master/docs/src/optimization_packages/optim.md
        #     Optim.KrylovTrustRegion(): A Newton-Krylov method with Trust Regions
        # initial_delta: The starting trust region radius
        # delta_hat: The largest allowable trust region radius
        # eta: When rho is at least eta, accept the step.
        # rho_lower: When rho is less than rho_lower, shrink the trust region.
        # rho_upper: When rho is greater than rho_upper, grow the trust region (though no greater than delta_hat).
        # Defaults:
        # initial_delta = 1.0  no should be initial_radius
        # delta_hat = 100.0 no max_radius
        # eta = 0.1
        # rho_lower = 0.25
        # rho_upper = 0.75
    # opt = Optimization.solve(fprob,
    #     Optim.KrylovTrustRegion(; initial_radius = 0.5, max_radius = 1000.0,
    #            eta = 0.1, rho_lower=0.25, rho_upper=500.),
    #   maxiters=maxiter, store_trace=true, show_trace=false)

    opt = Optimization.solve(fprob,
      Optim.KrylovTrustRegion(),
    maxiters=maxiter, store_trace=true, show_trace=false)
    # opt = Optimization.solve(fprob,
    #   Optim.KrylovTrustRegion(; initial_radius = 1.0, max_radius = 1000.0,
    #    eta = 0.1, rho_lower=0.101, rho_upper=0.75,
    #    cg_tol=0.01),
    #   maxiters=maxiter, store_trace=true, show_trace=false)

    result.solver_result = opt
    result.success = opt.retcode == Symbol("true")
    result.iterations = opt.original.iterations
    result.shares = opt.minimizer
    return result
end