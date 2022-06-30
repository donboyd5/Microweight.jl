function direct_cg(prob, result; shares0=fill(1. / prob.s, prob.h * prob.s), maxiter=100, objscale=1.0, interval=1, targweight, kwargs...)
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

    if targweight==nothing
       targweight = sqrt(length(shares0) / length(p_calctargets))
    end
    println("Target component weight: ", targweight)

    fp = (shares, p) -> objfn_direct(shares, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled,
        p_mshares, p_whs, p_calctargets, p_pdiffs, p_whpdiffs, interval, targweight)

    fpof = OptimizationFunction{true}(fp, Optimization.AutoZygote())
    fprob = OptimizationProblem(fpof, shares0, lb=zeros(length(shares0)), ub=ones(length(shares0)))

    # override default eta (0.4) but  use default linesearch
    opt = Optimization.solve(fprob,
    Optim.ConjugateGradient(
        # alphaguess https://github.com/JuliaNLSolvers/Optim.jl/blob/master/docs/src/algo/linesearch.md
        # LineSearches also allows the user to decide how the initial step length for the line search algorithm is chosen.
        # This is set with the alphaguess keyword argument for the Optim algorithm. The default procedure varies.
        # alphaguess = LineSearches.InitialStatic(), # default; best
        # alphaguess = LineSearches.InitialPrevious(), # NO
        # alphaguess = LineSearches.InitialQuadratic(), # NO
        #alphaguess = LineSearches.InitialConstantChange(ρ = 0.25), # fast, default
        # alphaguess = LineSearches.InitialConstantChange(ρ = 0.75), # best; ρ = 0.25 default
        # alphaguess = LineSearches.InitialHagerZhang(),
        # alphaguess = LineSearches.InitialHagerZhang(α0=1.0),
        # │   delta: Float64 0.1
        # │   sigma: Float64 0.9
        # │   alphamax: Float64 Inf
        # │   rho: Float64 5.0
        # │   epsilon: Float64 1.0e-6
        # │   gamma: Float64 0.66
        # │   linesearchmax: Int64 50
        # │   psi3: Float64 0.1
        # │   display: Int64 0
        # │   mayterminate: Base.RefValue{Bool}
        # │ , Flat()), NaN, 0.001, Optim.var"#49#51"())
        # linesearch = LineSearches.HagerZhang(), # best
        linesearch = LineSearches.HagerZhang(rho=5.0, psi3=0.1), # defaults seem best
        # linesearch = LineSearches.BackTracking(order=3), # 2nd best
        eta = 0.4 # best
        ), reltol=1e-24, maxiters=maxiter)

    # best for stub 9
    # opt = Optimization.solve(fprob,
    #     Optim.ConjugateGradient(
    #         # alphaguess https://github.com/JuliaNLSolvers/Optim.jl/blob/master/docs/src/algo/linesearch.md
    #         # LineSearches also allows the user to decide how the initial step length for the line search algorithm is chosen.
    #         # This is set with the alphaguess keyword argument for the Optim algorithm. The default procedure varies.
    #         # alphaguess = LineSearches.InitialStatic(), # default; best
    #         # alphaguess = LineSearches.InitialPrevious(), # NO
    #         # alphaguess = LineSearches.InitialQuadratic(), # NO
    #         # alphaguess = LineSearches.InitialConstantChange(ρ = 0.25), # fast, default
    #         alphaguess = LineSearches.InitialConstantChange(ρ = 0.75), # best; ρ = 0.25 default
    #         # alphaguess = LineSearches.InitialHagerZhang(),
    #         # alphaguess = LineSearches.InitialHagerZhang(α0=1.0),
    #         # │   delta: Float64 0.1
    #         # │   sigma: Float64 0.9
    #         # │   alphamax: Float64 Inf
    #         # │   rho: Float64 5.0
    #         # │   epsilon: Float64 1.0e-6
    #         # │   gamma: Float64 0.66
    #         # │   linesearchmax: Int64 50
    #         # │   psi3: Float64 0.1
    #         # │   display: Int64 0
    #         # │   mayterminate: Base.RefValue{Bool}
    #         # │ , Flat()), NaN, 0.001, Optim.var"#49#51"())
    #         # linesearch = LineSearches.HagerZhang(), # best
    #         linesearch = LineSearches.HagerZhang(rho=5.0, psi3=0.1), # defaults seem best
    #         # linesearch = LineSearches.BackTracking(order=3), # 2nd best
    #         eta = 0.7 # best
    #         ), reltol=1e-24, maxiters=maxiter)

    result.solver_result = opt
    result.success = opt.retcode == Symbol("true") || (opt.original.iterations >= maxiter)
    result.iterations = opt.original.iterations
    result.shares = opt.minimizer
    return result
end