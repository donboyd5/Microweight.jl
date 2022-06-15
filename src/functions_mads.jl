

function mads(prob, beta0, result; maxiter=100, objscale, kwargs...)
    # for allowable arguments:
    # https://github.com/JuliaNLSolvers/LsqFit.jl/blob/master/src/levenberg_marquardt.jl
    kwkeys_allowed = (:show_trace, :x_tol, :g_tol)
    kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)

    # for details on Mads.levenberg_marquardt
    #   http://madsjulia.github.io/Mads.jl/Modules/Mads/#Mads.levenberg_marquardt

    f = beta -> objvec(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled) .* objscale
    # function gvec(beta)
    #     ForwardDiff.jacobian(x -> fvec(x), beta)
    # end
    g = beta -> ForwardDiff.jacobian(beta -> objvec(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled) .* objscale, beta)

    function callback(x_best::AbstractVector, of::Number, lambda::Number)
        global callbacksucceeded
        callbacksucceeded = true
        println(of, " ", lambda)
    end
    # defaults
    # opt = Mads.levenberg_marquardt(f, g, beta0,
    #   lambda_scale=1e-3, lambda_mu=10, lambda_nu=2, np_lambda=10,
    #   maxIter=100, maxJacobians=100, maxEval=1001,
    #   tolOF=1e-3, tolX=1e-4, tolG=1e-6, callbackiteration=callback)
    # lambda_mu=0.1, np_lambda=10,  maxJacobians=10000, maxEval=1000000,
    opt = Mads.levenberg_marquardt(f, g, beta0,
      lambda_scale=1e-3, lambda_mu=10, lambda_nu=2, np_lambda=10,
      maxIter=200, maxJacobians=10000, maxEval=1000000,
      tolOF=1e-3, tolX=1e-8, tolG=1e-6, callbackiteration=callback)

    # result.success = false
    result.success = opt.iteration_converged || opt.x_converged || opt.f_converged || opt.g_converged # include f_converged??
    result.iterations = opt.iterations
    result.beta = opt.minimizer
    result.solver_result = opt

    return result
end