#=
https://github.com/matthieugomez/LeastSquaresOptim.jl
https://github.com/JuliaNLSolvers/LsqFit.jl/blob/master/src/levenberg_marquardt.jl


=#

function poisson_lsoptim(prob, result; maxiter=100, objscale, interval=1, kwargs...)
    # for allowable arguments:
    # https://github.com/JuliaNLSolvers/LsqFit.jl/blob/master/src/levenberg_marquardt.jl
    kwkeys_allowed = (:show_trace, :x_tol, :g_tol)
    kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)

    f = beta -> objvec2(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled, interval) # .* objscale

    opt = LeastSquaresOptim.optimize(f, result.beta0, LevenbergMarquardt(LeastSquaresOptim.LSMR()),
      autodiff = :forward, show_trace=false, iterations=maxiter)

    result.success = opt.converged
    result.iterations = opt.iterations
    result.beta = opt.minimizer
    result.solver_result = opt

    return result
end