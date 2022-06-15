#=
Options


=#

function cg_optim(prob, beta0, result; maxiter=100, objscale, kwargs...)
    # for allowable arguments:
    # https://github.com/JuliaNLSolvers/LsqFit.jl/blob/master/src/levenberg_marquardt.jl
    kwkeys_allowed = (:show_trace, :x_tol, :g_tol)
    kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)

    f = beta -> objfn(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled) .* objscale

    opt = Optim.optimize(f, beta0, ConjugateGradient(),
      Optim.Options(g_tol = 1e-6, iterations = maxiter, store_trace = true, show_trace = true);
      autodiff = :forward)

    result.solver_result = opt
    result.success = opt.iteration_converged || opt.x_converged || opt.f_converged || opt.g_converged
    result.iterations = opt.iterations
    result.beta = opt.minimizer

    return result
end