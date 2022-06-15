
function minpack(prob, beta0, result; maxiter=100, kwargs...)
    # for allowable arguments:
    # https://github.com/JuliaNLSolvers/LsqFit.jl/blob/master/src/levenberg_marquardt.jl
    kwkeys_allowed = (:show_trace, :x_tol, :g_tol)
    kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)

    # MUST have in-place functions with arguments (out, beta) [or other names with same meanings]
    # f! = (out, beta) -> objvec!(out, beta, prob.wh, prob.xmat, prob.geotargets)
    f! = (out, beta) -> out .= objvec(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled)
    g! = (out, beta) -> out .= ForwardDiff.jacobian(beta -> objvec(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled), beta)

    println("NOTE: MINPACK's trace shows f(x) inf-norm, which is max(abs(% diff from target))")
    opt = fsolve(f!, g!, beta0, show_trace=true, method=:lm, ftol=1e-6, xtol=1e-6, iterations=maxiter)

    result.success = opt.converged
    result.iterations = opt.trace.f_calls
    result.beta = opt.x
    result.solver_result = opt

    return result
end