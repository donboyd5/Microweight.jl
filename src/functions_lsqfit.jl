

function lsqlm(ibeta, prob, result; maxiter=100, kwargs...)
    kwkeys_allowed = (:show_trace, :x_tol, :g_tol)
    kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)

    f = beta -> objvec(beta, prob.wh, prob.xmat, prob.geotargets)
    f_init = f(ibeta)
    od = OnceDifferentiable(f, ibeta, copy(f_init); inplace = false, autodiff = :forward)
    opt = LsqFit.levenberg_marquardt(od, ibeta; maxIter=maxiter, kwargs_keep...)

    result.success = opt.iteration_converged || opt.x_converged || opt.g_converged
    result.iterations = opt.iterations
    result.sspd = opt.minimum
    result.beta = opt.minimizer
    result.solver_result = opt

    return result
    # LsqFit.lmfit(f10, x, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=50)
end