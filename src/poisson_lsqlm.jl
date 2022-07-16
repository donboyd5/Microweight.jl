

#=
https://github.com/JuliaNLSolvers/LsqFit.jl/blob/master/src/levenberg_marquardt.jl

# Keyword arguments
* `x_tol::Real=1e-8`: search tolerance in x
* `g_tol::Real=1e-12`: search tolerance in gradient
* `maxIter::Integer=1000`: maximum number of iterations
* `min_step_quality=1e-3`: for steps below this quality, the trust region is shrinked
* `good_step_quality=0.75`: for steps above this quality, the trust region is expanded
* `lambda::Real=10`: (inverse of) initial trust region radius
* `tau=Inf`: set initial trust region radius using the heuristic : tau*maximum(jacobian(df)'*jacobian(df))
* `lambda_increase=10.0`: `lambda` is multiplied by this factor after step below min quality
* `lambda_decrease=0.1`: `lambda` is multiplied by this factor after good quality steps
* `show_trace::Bool=false`: print a status summary on each iteration if true
* `lower,upper=[]`: bound solution to these limits

don't pass maxIter
(:x_tol, :g_tol, :min_step_quality, :good_step_quality, :lambda, :tau, :lambda_increase, :lambda_decrease, :show_trace, :lower, :upper)

=#

function poisson_lsqlm(prob, result; maxiter=100, objscale=1, kwargs...)
    # for allowable arguments:
    # https://github.com/JuliaNLSolvers/LsqFit.jl/blob/master/src/levenberg_marquardt.jl

    println("kwargs requested: ", keys(kwargs))
    kwkeys_allowed = (:x_tol, :g_tol, :min_step_quality, :good_step_quality, :lambda, :tau, :lambda_increase, :lambda_decrease, :show_trace, :lower, :upper)
    println("kwargs allowed: ", kwkeys_allowed)
    kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)
    println("kwargs passed on: $kwargs_keep")

    f = beta -> objvec_poisson(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled) .* objscale
    f_init = f(result.beta0)
    od = NLSolversBase.OnceDifferentiable(f, result.beta0, copy(f_init); inplace = false, autodiff = :forward)
    opt = LsqFit.levenberg_marquardt(od, result.beta0; maxIter=maxiter, kwargs_keep...)

    result.solver_result = opt
    result.success = opt.iteration_converged || opt.x_converged || opt.f_converged || opt.g_converged # include f_converged??
    result.iterations = opt.iterations
    result.beta = opt.minimizer

    return result
end