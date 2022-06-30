#=
Notes
https://discourse.julialang.org/t/convergence-in-optim-questions/65139
      once_diff = OnceDifferentiable(f, [1.98,-1.95]; autodiff = :forward);
res = Optim.optimize(once_diff, [1.25, -2.1], [Inf, Inf], [2.0, 2.0], Fminbox(ConjugateGradient()),
                     Optim.Options(x_abstol = 1e-3, x_reltol = 1e-3, f_abstol = 1e-3, f_reltol =
                                   1e-3, g_tol = 1e-3))

Options


=#

function poisson_cgoptim2(prob, beta0, result; maxiter=100, objscale, interval, kwargs...)
    # for allowable arguments:

    kwkeys_allowed = (:show_trace, :x_tol, :g_tol)
    kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)

    # f = beta -> objfn(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled) .* objscale
    # f = beta -> objfn2(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled, interval) .* objscale


    fp = (beta, p) -> objfn2(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled, interval) .* objscale

    fpof = OptimizationFunction{true}(fp, Optimization.AutoZygote())
    fprob = OptimizationProblem(fpof, beta0)

    # override default eta (0.4) but  use default linesearch
    # opt = Optimization.solve(fprob, Optim.ConjugateGradient(alphaguess = linesearch = LineSearches.HagerZhang(), eta = 0.4), maxiters=maxiter)
    opt = Optimization.solve(fprob,
            Optim.ConjugateGradient(
            # Optim.GradientDescent(
            # Optim.LBFGS(
                # alphaguess = LineSearches.InitialStatic(), # NO; default
                # alphaguess = LineSearches.InitialPrevious(), # NO
                # alphaguess = LineSearches.InitialQuadratic(), # NO
                # alphaguess = LineSearches.InitialConstantChange(ρ = 0.05), # NO ; ρ = 0.25 default
                # alphaguess = LineSearches.InitialHagerZhang(),
                alphaguess = LineSearches.InitialHagerZhang(α0=1.0),
                # linesearch = LineSearches.HagerZhang(), # default
                linesearch = LineSearches.BackTracking(order=3), # best
                # linesearch = LineSearches.MoreThuente(), # NO
                # linesearch = LineSearches.Static(), # NO
                # linesearch = LineSearches.StrongWolfe(), # NO
                # linesearch = LineSearches.MoreThuente(),
                eta = 0.4  # 0.4 defaults
            ),
            maxiters=maxiter)

    # opt = Optim.optimize(od, beta0,
    #   Optim.ConjugateGradient(eta=0.01; alphaguess = LineSearches.InitialConstantChange(), linesearch = LineSearches.HagerZhang()),
    #   Optim.Options(x_abstol = 1e-8, x_reltol = 1e-8, f_abstol = 1e-8, f_reltol =1e-8, g_tol = 0.,
    #                 iterations = maxiter, store_trace = true, show_trace = false))

    result.solver_result = opt
    result.success = opt.retcode == Symbol("true")
    result.iterations = opt.original.iterations
    result.beta = opt.minimizer

    return result
end