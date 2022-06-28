#=
Options


=#

function poisson_newttrust(prob, beta0, result; maxiter=100, objscale, interval=1, kwargs...)
    # for allowable arguments:
    # https://github.com/JuliaNLSolvers/LsqFit.jl/blob/master/src/levenberg_marquardt.jl
    kwkeys_allowed = (:show_trace, :x_tol, :g_tol)
    kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)

    global fcalls

    # f = beta -> objvec(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled) .* objscale
    # f = beta -> objvec2(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled, fcalls, interval) .* objscale
    f! = (out, beta) -> out .= objvec2(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled, fcalls, interval) .* objscale

    # opt = LeastSquaresOptim.optimize(f, beta0, LevenbergMarquardt(LeastSquaresOptim.LSMR()),
    #   autodiff = :forward, show_trace=true, iterations=maxiter)
    # factor default is 1.0
    opt = NLsolve.nlsolve(f!, beta0, autodiff=:forward, autoscale=true, factor=1.0, method = :trust_region, iterations=maxiter, show_trace = false)
    # opt = NLsolve.nlsolve(f!, beta0, autodiff=:forward, method = :anderson, m=100, iterations=maxiter, show_trace = false)
    # defaults

    result.solver_result = opt
    result.success = opt.x_converged || opt.f_converged
    result.iterations = opt.iterations
    result.beta = opt.zero

    return result
end