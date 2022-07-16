#=
https://github.com/JuliaNLSolvers/NLsolve.jl

Other optional arguments to nlsolve, available for all algorithms, are:
    xtol: norm difference in x between two successive iterates under which convergence is declared. Default: 0.0.
    ftol: infinite norm of residuals under which convergence is declared. Default: 1e-8.
    iterations: maximum number of iterations. Default: 1_000.
    store_trace: should a trace of the optimization algorithm's state be stored? Default: false.
    show_trace: should a trace of the optimization algorithm's state be shown on STDOUT? Default: false.
    extended_trace: should additifonal algorithm internals be added to the state trace? Default: false.

    don't use iterations
    (:xtol, :ftol, :store_trace, :show_trace, :extended_trace)

=#

function poisson_nlsolve(prob, result; maxiter=100, objscale, kwargs...)
    # for allowable arguments:
    # https://github.com/JuliaNLSolvers/LsqFit.jl/blob/master/src/levenberg_marquardt.jl

    f! = (out, beta) -> out .= objvec_poisson(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled) .* objscale
    # j! = (out, beta) -> out .= ForwardDiff.jacobian(beta ->
    #     objvec_poisson(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled) .* objscale, beta)

    method = result.method
    if method==:newttr_nlsolve algorithm=:trust_region
    # elseif method==:lbfgs_nlopt algorithm=:(LD_LBFGS())
    # elseif method==:mma algorithm=:(LD_MMA())
    # elseif method==:newton algorithm=:(LD_TNEWTON_PRECOND())
    # elseif method==:newtonrs algorithm=:(LD_TNEWTON_PRECOND_RESTART())
    # elseif method==:var1 algorithm=:(LD_VAR1())
    # elseif method==:var2 algorithm=:(LD_VAR2())
    else return "ERROR: method must be one of (:ccsaq, :lbfgs, :mma, :newton, :var2)"
    end
    println("NLsolve algorithm: ", algorithm)

    println("kwargs requested: ", keys(kwargs))
    kwkeys_method = (:xtol, :ftol, :store_trace, :show_trace, :extended_trace)
    kwkeys_algo = (:factor, :autoscale)
    # merge the allowable sets of keys
    kwkeys_allowed = (kwkeys_method..., kwkeys_algo...)
    println("kwargs allowed: ", kwkeys_allowed)
    kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)
    println("kwargs passed on: $kwargs_keep")

    opt = NLsolve.nlsolve(f!, result.beta0, autodiff=:forward, method = algorithm,
            iterations=maxiter; kwargs_keep...)
    # opt = NLsolve.nlsolve(f!, beta0, autodiff=:forward, method = :anderson, m=100, iterations=maxiter, show_trace = false)

    result.solver_result = opt
    result.success = opt.x_converged || opt.f_converged
    result.iterations = opt.iterations
    result.beta = opt.zero

    return result
end