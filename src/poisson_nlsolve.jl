#=
https://github.com/JuliaNLSolvers/NLsolve.jl

Other optional arguments to nlsolve, available for all algorithms, are:
    - xtol: norm difference in x between two successive iterates under which convergence is declared. Default: 0.0.
    - ftol: infinite norm of residuals under which convergence is declared. Default: 1e-8.
    - iterations: maximum number of iterations. Default: 1_000.
    - store_trace: should a trace of the optimization algorithm's state be stored? Default: false.
    - show_trace: should a trace of the optimization algorithm's state be shown on STDOUT? Default: false.
     extended_trace: should additifonal algorithm internals be added to the state trace? Default: false.

    don't use iterations
    (:xtol, :ftol, :store_trace, :show_trace, :extended_trace)

The following methods are available:
    (:anderson, :broyden, :newton, :trust_region) - I map these to:
    (:anderson, :broyden, :newton_nlsolve, :trust_nlsolve)

    :anderson   Anderson acceleration
    Also known as DIIS or Pulay mixing, this method is based on the acceleration of the fixed-point iteration xₙ₊₁ = xₙ + beta*f(xₙ),
    where by default beta=1. It does not use Jacobian information or linesearch, but has a history whose size is controlled by
    the m parameter: m=0 (the default) corresponds to the simple fixed-point iteration above, and higher values use a larger
    history size to accelerate the iterations. Higher values of m usually increase the speed of convergence, but increase
    the storage and computation requirements and might lead to instabilities.
      test local convergence of Anderson: close to a fixed-point and with a small beta, f should be almost affine, in which
      case Anderson is equivalent to GMRES and should converge
        nlsolve(df, [ 0.01; .99], method = :anderson, m = 10, beta=.01)
    (:m, :beta)

    :broyden same options as newton I think

    :newton   Newton method with linesearch
    - linesearch, which must be equal to a function computing the linesearch. Currently, available values are taken
      from the LineSearches package. By default, no linesearch is performed.
      LineSearches.BackTracking(), LineSearches.HagerZhang(), LineSearches.StrongWolfe()
    (:linesearch,)

    :trust_region
    - factor: determines the size of the initial trust region. This size is set to the product of factor
      and the euclidean norm of initial_x if nonzero, or else to factor itself. Default: 1.0.
    - autoscale: if true, then the variables will be automatically rescaled. The scaling factors are the norms
      of the Jacobian columns. Default: true.
    (:factor, :autoscale)

Defaults
    xtol::Real = zero(real(eltype(initial_x))),
    ftol::Real = convert(real(eltype(initial_x)), 1e-8),
    iterations::Integer = 1_000,
    store_trace::Bool = false,
    show_trace::Bool = false,
    extended_trace::Bool = false,
    linesearch = LineSearches.Static(),
    linsolve=(x, A, b) -> copyto!(x, A\b),
    factor::Real = one(real(eltype(initial_x))),
    autoscale::Bool = true,
    m::Integer = 10,
    beta::Real = 1,
    aa_start::Integer = 1,
    droptol::Real = convert(real(eltype(initial_x)), 1e10))

=#

function poisson_nlsolve(prob, result; maxiter=100, objscale, kwargs...)
    f! = (out, beta) -> out .= objvec_poisson(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled) .* objscale
    # function f!(out, beta)
    #     out = objvec_poisson(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled) # .* objscale
    #     out
    # end

    # (:anderson, :broyden, :newton_nlsolve, :trust_nlsolve) my key words, map these to
    # (:anderson, :broyden, :newton, :trust_region) the nlsolve key words
    # summary of allowed options:
    #   anderson: m, beta
    #   broyden: linesearch - LineSearches.Static() default
    #   newton_nlsolve: linesearch - same
    #   trust_nlsolve: autoscale, factor
    method = result.method
    if method==:anderson
        println("WARNING: anderson throws ERROR: LinearAlgebra.LAPACKException(1)")
        algorithm=:anderson
        kwkeys_algo=(:m, :beta)
    elseif method==:broyden
        println("WARNING: broyden does not seem to return good results - examine iteration results")
        algorithm=:broyden
        kwkeys_algo=(:linesearch,)
    elseif method==:newton_nlsolve
        algorithm=:newton
        kwkeys_algo=(:linesearch,)
    elseif method==:trust_nlsolve
        algorithm=:trust_region
        kwkeys_algo=(:autoscale, :factor)
    else return "ERROR: method must be one of (:anderson, :broyden, :newton_nlsolve, :trust_nlsolve)"
    end
    println("NLsolve algorithm: ", algorithm)

    kwkeys_method = (:xtol, :ftol, :store_trace, :show_trace, :extended_trace)
    # kwkeys_algo = NamedTuple()  use kwkeys_algo from above
    kwargs_defaults = Dict() # :stopval => 1e-4
    kwargs_use = kwargs_keep(kwargs; kwkeys_method=kwkeys_method, kwkeys_algo=kwkeys_algo, kwargs_defaults=kwargs_defaults)

    opt = NLsolve.nlsolve(f!, result.beta0, autodiff=:forward, method = algorithm,
            iterations=maxiter; kwargs_use...)
    # opt = NLsolve.nlsolve(f!, beta0, autodiff=:forward, method = :anderson, m=100, iterations=maxiter, show_trace = false)

    result.solver_result = opt
    result.success = opt.x_converged || opt.f_converged
    result.iterations = opt.iterations
    result.beta = opt.zero

    return result
end