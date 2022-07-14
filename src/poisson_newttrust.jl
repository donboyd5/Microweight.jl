#=
https://github.com/JuliaNLSolvers/NLsolve.jl

Options


=#

function poisson_newttrust(prob, result; maxiter=100, objscale, kwargs...)
    # for allowable arguments:
    # https://github.com/JuliaNLSolvers/LsqFit.jl/blob/master/src/levenberg_marquardt.jl
    # kwkeys_allowed = (:show_trace, :x_tol, :g_tol)
    kwkeys_allowed = (:factor, :autoscale, :xtol, :ftol)
    kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)


    f! = (out, beta) -> out .= objvec_poisson(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled) .* objscale
    # j! = (out, beta) -> out .= ForwardDiff.jacobian(beta ->
    #     objvec_poisson(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled) .* objscale, beta)

    # opt = LeastSquaresOptim.optimize(f, beta0, LevenbergMarquardt(LeastSquaresOptim.LSMR()),
    #   autodiff = :forward, show_trace=true, iterations=maxiter)
    # factor default is 1.0
    opt = NLsolve.nlsolve(f!, result.beta0, autodiff=:forward, method = :trust_region,
            iterations=maxiter, show_trace = false; kwargs_keep...)
    # opt = NLsolve.nlsolve(f!, beta0, autodiff=:forward, method = :anderson, m=100, iterations=maxiter, show_trace = false)
    # defaults

    result.solver_result = opt
    result.success = opt.x_converged || opt.f_converged
    result.iterations = opt.iterations
    result.beta = opt.zero

    return result
end