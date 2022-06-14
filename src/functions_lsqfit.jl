

function lsqlm(prob, beta0, result; maxiter=100, kwargs...)
    # for allowable arguments:
    # https://github.com/JuliaNLSolvers/LsqFit.jl/blob/master/src/levenberg_marquardt.jl
    kwkeys_allowed = (:show_trace, :x_tol, :g_tol)
    kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)

    f = beta -> objvec(beta, prob.wh, prob.xmat, prob.geotargets)
    f_init = f(beta0)
    od = OnceDifferentiable(f, beta0, copy(f_init); inplace = false, autodiff = :forward)
    opt = LsqFit.levenberg_marquardt(od, beta0; maxIter=maxiter, kwargs_keep...)

    result.success = opt.iteration_converged || opt.x_converged || opt.g_converged
    result.iterations = opt.iterations
    result.sspd = opt.minimum
    result.beta = opt.minimizer
    result.solver_result = opt

    return result
end