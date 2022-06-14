
# function fvec!(out, beta)
#     # for LeastSquaresOptim inplace
#     out .= objvec(beta, wh, xmat, geotargets)
# end

# function gvec!(out, beta)
#     out .= ForwardDiff.jacobian(x -> fvec(x), beta)
# end

# function fvec(beta)
#     # beta = reshape(beta, size(geotargets))
#     objvec(beta, wh, xmat, geotargets)
#   end


function minpack(prob, beta0, result; maxiter=100, kwargs...)
    # for allowable arguments:
    # https://github.com/JuliaNLSolvers/LsqFit.jl/blob/master/src/levenberg_marquardt.jl
    kwkeys_allowed = (:show_trace, :x_tol, :g_tol)
    kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)

    # f = beta -> objvec(beta, prob.wh, prob.xmat, prob.geotargets)
    # f_init = f(beta0)
    # od = OnceDifferentiable(f, beta0, copy(f_init); inplace = false, autodiff = :forward)
    # opt = LsqFit.levenberg_marquardt(od, beta0; maxIter=maxiter, kwargs_keep...)
    # fvec = beta -> objvec(beta, prob.wh, prob.xmat, prob.geotargets)
    fvec! = (out, beta) -> objvec!(out, beta, prob.wh, prob.xmat, prob.geotargets)
    # fvec! = beta -> objvec!(out, beta, prob.wh, prob.xmat, prob.geotargets)

    function gvec!(out, beta)
        # out .= ForwardDiff.jacobian(beta -> fvec(beta), beta)
        out .= ForwardDiff.jacobian(beta -> objvec(beta, prob.wh, prob.xmat, prob.geotargets), beta)
        out
    end

    # g! = beta -> gvec2!(out, beta, prob)

    # println("defined gvec!")
    # out = similar(beta0)
    # fvec!(beta0)
    # out = zeros(length(beta0), length(beta0))
    # println(g!(beta0))
    # println("called gvec!")
    # return

    opt = fsolve(fvec!, gvec!, beta0, show_trace=true, method=:lm, ftol=1e-6, xtol=1e-6, iterations=20)
    # opt = fsolve(fvec!, beta0, show_trace=true, method=:lm, ftol=1e-6, xtol=1e-6, iterations=20)
    println("called fsolve")

    result.success = false
    # result.iterations = opt.iterations
    # result.sspd = opt.minimum
    # result.beta = opt.minimizer
    result.solver_result = opt

    return result
end