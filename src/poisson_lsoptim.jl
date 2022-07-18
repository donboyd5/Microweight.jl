#=
https://github.com/matthieugomez/LeastSquaresOptim.jl
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

# don't include maxIter
(:x_tol, :g_tol, :min_step_quality, :good_step_quality, :lambda, :tau, :lambda_increase, :lambda_decrease, :show_trace, :lower, :upper)


=#

function poisson_lsoptim(prob, result; maxiter=100, objscale, interval=1, kwargs...)
    kwkeys_method = (:x_tol, :g_tol, :min_step_quality, :good_step_quality, :lambda, :tau, :lambda_increase, :lambda_decrease, :show_trace, :lower, :upper)
    kwkeys_algo = NamedTuple()
    kwargs_defaults = Dict() # :stopval => 1e-4
    kwargs_use = kwargs_keep(kwargs; kwkeys_method=kwkeys_method, kwkeys_algo=kwkeys_algo, kwargs_defaults=kwargs_defaults)

    f = beta -> objvec_poisson(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled) .* objscale

    opt = LeastSquaresOptim.optimize(f, result.beta0, LevenbergMarquardt(LeastSquaresOptim.LSMR()),
      autodiff = :forward, iterations=maxiter; kwargs_use...)

    result.success = opt.converged
    result.iterations = opt.iterations
    result.beta = opt.minimizer
    result.solver_result = opt

    return result
end