#=
https://github.com/matthieugomez/LeastSquaresOptim.jl


=#

function lsoptim(prob, beta0, result; maxiter=100, objscale, kwargs...)
    # for allowable arguments:
    # https://github.com/JuliaNLSolvers/LsqFit.jl/blob/master/src/levenberg_marquardt.jl
    kwkeys_allowed = (:show_trace, :x_tol, :g_tol)
    kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)

    f = beta -> objvec(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled) .* objscale

    opt = LeastSquaresOptim.optimize(f, beta0, LevenbergMarquardt(LeastSquaresOptim.LSMR()),
      autodiff = :forward, show_trace=true, iterations=maxiter)
    # defaults

    result.success = false
    result.success = opt.converged
    result.iterations = opt.iterations
    result.beta = opt.minimizer
    result.solver_result = opt

    return result
end