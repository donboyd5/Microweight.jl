#=
Notes
https://discourse.julialang.org/t/convergence-in-optim-questions/65139
      once_diff = OnceDifferentiable(f, [1.98,-1.95]; autodiff = :forward);
res = Optim.optimize(once_diff, [1.25, -2.1], [Inf, Inf], [2.0, 2.0], Fminbox(ConjugateGradient()),
                     Optim.Options(x_abstol = 1e-3, x_reltol = 1e-3, f_abstol = 1e-3, f_reltol =
                                   1e-3, g_tol = 1e-3))

Options


=#

function poisson_optz_optimisers(prob, result; maxiter=1000, objscale, pow, targstop, whstop,
      kwargs...)

      fp = (beta, p) -> objfn_poisson(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled, pow, targstop, whstop) .* objscale
      fpof = OptimizationFunction{true}(fp, Optimization.AutoZygote())
      fprob = OptimizationProblem(fpof, result.beta0)

      method = result.method
      if method==:adam algorithm=:(Adam(0.0001, (.9, .999)))  # :(Adam(0.5))
    #   elseif method==:lbfgs_optim algorithm=:(LBFGS(; m=100))
      elseif method==:descent algorithm=:(Descent())
      elseif method==:momentum algorithm=:(Momentum())
      elseif method==:nesterov algorithm=:(Nesterov(0.00001, 0.9))
      else return "ERROR: method must be one of (:adam,  :descent, :momentum, :nesterov)"
      end

      kwkeys_method = (:maxtime, :abstol, :reltol)
      kwkeys_algo = NamedTuple()
      kwargs_defaults = Dict() # :stopval => 1e-4
      kwargs_use = kwargs_keep(kwargs; kwkeys_method=kwkeys_method, kwkeys_algo=kwkeys_algo, kwargs_defaults=kwargs_defaults)

      println("Optim algorithm: ", algorithm)
      opt = Optimization.solve(fprob,
        Optimisers.eval(algorithm), maxiters=maxiter, callback=cb_poisson; kwargs_use...) # , callback=cb_poisson

      result.solver_result = opt
      result.success = true
      result.iterations = -999 # opt.original.iterations
      result.beta = opt.minimizer

      return result
  end

# ALTERNATIVE FORMULATIONS THAT MAY BE PROMISING

#   opt = Optimization.solve(fprob,
#             Optim.ConjugateGradient(
#             # Optim.GradientDescent(
#             # Optim.LBFGS(
#                 # alphaguess = LineSearches.InitialStatic(), # NO; default
#                 # alphaguess = LineSearches.InitialPrevious(), # NO
#                 # alphaguess = LineSearches.InitialQuadratic(), # NO
#                 # alphaguess = LineSearches.InitialConstantChange(ρ = 0.05), # NO ; ρ = 0.25 default
#                 # alphaguess = LineSearches.InitialHagerZhang(),
#                 alphaguess = LineSearches.InitialHagerZhang(α0=1.0),
#                 # linesearch = LineSearches.HagerZhang(), # default
#                 linesearch = LineSearches.BackTracking(order=3), # best
#                 # linesearch = LineSearches.MoreThuente(), # NO
#                 # linesearch = LineSearches.Static(), # NO
#                 # linesearch = LineSearches.StrongWolfe(), # NO
#                 # linesearch = LineSearches.MoreThuente(),
#                 eta = 0.4  # 0.4 defaults
#             ),
#             maxiters=maxiter)

    # opt = Optim.optimize(od, beta0,
    #   Optim.ConjugateGradient(eta=0.01; alphaguess = LineSearches.InitialConstantChange(), linesearch = LineSearches.HagerZhang()),
    #   Optim.Options(x_abstol = 1e-8, x_reltol = 1e-8, f_abstol = 1e-8, f_reltol =1e-8, g_tol = 0.,
    #                 iterations = maxiter, store_trace = true, show_trace = false))


