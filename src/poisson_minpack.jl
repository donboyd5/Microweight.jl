
# https://github.com/sglyon/MINPACK.jl
# Julia interface to cminpack, a C/C++ rewrite of the MINPACK software (originally in fortran).

#=

lm and hybr are available when passing f and g

Options for :lm (based on lmder):
https://github.com/devernay/cminpack/blob/d1f5f5a273862ca1bbcf58394e4ac060d9e22c76/lmder.c
    ftol is a nonnegative input variable. termination
        occurs when both the actual and predicted relative
        reductions in the sum of squares are at most ftol.
        therefore, ftol measures the relative error desired
        in the sum of squares.

    xtol is a nonnegative input variable. termination
        occurs when the relative error between two consecutive
        iterates is at most xtol. therefore, xtol measures the
        relative error desired in the approximate solution.

    gtol is a nonnegative input variable. termination
        occurs when the cosine of the angle between fvec and
        any column of the jacobian is at most gtol in absolute
        value. therefore, gtol measures the orthogonality
        desired between the function vector and the columns
        of the jacobian.

    maxfev is a positive integer input variable. termination
        occurs when the number of calls to fcn with iflag = 1
        has reached maxfev.

=#

function poisson_minpack(prob, result; maxiter=1000, objscale, kwargs...)
    # for allowable arguments:
    # lm: https://github.com/JuliaNLSolvers/LsqFit.jl/blob/master/src/levenberg_marquardt.jl
    # hybr: # maxfev, epsfcn, diag, mode, factor, nprint, lr
    kwkeys_allowed = (:factor, :xtol, :mode)
    kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)
    # println("kwargs: X", kwargs)
    # println("kwargs_keep: X", kwargs_keep)
    # return

    # MUST have in-place functions with arguments (out, beta) [or other names with same meanings]
    # f! = (out, beta) -> objvec!(out, beta, prob.wh, prob.xmat, prob.geotargets)
    # f! = (out, beta) -> out .= objvec(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled) .* objscale
    # g! = (out, beta) -> out .= ForwardDiff.jacobian(beta -> objvec(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled) .* objscale, beta)

    f! = (out, beta) -> out .= objvec_poisson(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled) .* objscale
    g! = (out, beta) -> out .= ForwardDiff.jacobian(beta ->
        objvec_poisson(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled) .* objscale, beta)

    # println("NOTE: MINPACK's trace shows f(x) inf-norm, which is max(abs(% diff from target))")
    # opt = MINPACK.fsolve(f!, g!, result.beta0, show_trace=false, method=:lm, ftol=1e-6, xtol=1e-6, iterations=maxiter)
    if result.method==:lm_minpack algorithm=:lm
    elseif result.method==:hybr_minpack algorithm=:hybr
    else return "ERROR: method must be one of (hybr_minpack, :lm_minpack)"
    end
    println("algorithm: ", algorithm)

    # maxfev, epsfcn, diag, mode, factor, nprint, lr
    # opt = MINPACK.fsolve(f!, g!, result.beta0, show_trace=false, save_trace=true, method=algorithm, iterations=maxiter, kwargs...)
    # note the semicolon below!!
    opt = MINPACK.fsolve(f!, g!, result.beta0, method=:hybr, iterations=maxiter; kwargs_keep...)

    result.success = true
    result.iterations = opt.trace.f_calls
    result.beta = opt.x
    result.solver_result = opt

    return result
end