#=
Notes
https://discourse.julialang.org/t/convergence-in-optim-questions/65139
      once_diff = OnceDifferentiable(f, [1.98,-1.95]; autodiff = :forward);
res = Optim.optimize(once_diff, [1.25, -2.1], [Inf, Inf], [2.0, 2.0], Fminbox(ConjugateGradient()),
                     Optim.Options(x_abstol = 1e-3, x_reltol = 1e-3, f_abstol = 1e-3, f_reltol =
                                   1e-3, g_tol = 1e-3))

Options


=#

function poisson_optim(prob, result; maxiter=100, objscale, pow, targstop, whstop,
      kwargs...)
      # for allowable arguments:

      kwkeys_allowed = (:show_trace, :x_tol, :g_tol)
      kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)

      fp = (beta, p) -> objfn_poisson(beta, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled, pow, targstop, whstop, objscale)
      fpof = OptimizationFunction{true}(fp, Optimization.AutoZygote())
      fprob = OptimizationProblem(fpof, result.beta0)

      method = result.method
      if method==:cg algorithm=:(ConjugateGradient())
      elseif method==:gd algorithm=:(GradientDescent())
      elseif method==:lbfgs_optim algorithm=:(LBFGS())
      elseif method==:krylov algorithm=:(KrylovTrustRegion())
      else return "ERROR: method must be one of (:cg, gd, :lbfgs_optim, krylov)"
      end

      println("Optim algorithm: ", algorithm)
      opt = Optimization.solve(fprob,
            Optim.eval(algorithm), maxiters=maxiter, callback=cb_poisson) # , callback=cb_poisson

      result.solver_result = opt
      result.success = true
      result.iterations = opt.original.iterations
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


