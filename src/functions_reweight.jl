
# module BruteForce

# https://docs.sciml.ai/Optimization/stable/API/optimization_function/
# AutoForwardDiff(): The fastest choice for small optimizations
# AutoReverseDiff(compile=false): A fast choice for large scalar optimizations
# AutoTracker(): Like ReverseDiff but GPU-compatible
# AutoZygote(): The fastest choice for non-mutating array-based (BLAS) functions
# AutoFiniteDiff(): Finite differencing, not optimal but always applicable
# AutoModelingToolkit(): The fastest choice for large scalar optimizations


# %% utility functions

function rwscale(xmat, rwtargets)
  # scale xmat so that mean value is 1.0, and scale rwtargets accordingly
  scale = vec(sum(abs.(xmat), dims=1)) ./ size(xmat)[1] 
  # scale = fill(1., k) # unscaled
  xmat = xmat ./ scale' 
  rwtargets = rwtargets ./ scale
  # mean(abs.(xmat), dims=1)
  return xmat, rwtargets
end

# %% opt functions

function objfn_reweight(
  ratio, wh, xmat, rwtargets;
  rweight=0.1, # relative importance of minimizing ratio error rather than target error
  targstop=true, whstop=true,
  display_progress=true)

  # part 1 get measure of difference from targets
  rwtargets_calc = xmat' * (ratio .* wh)
  targpdiffs = (rwtargets_calc .- rwtargets) ./ rwtargets # ./ 1e6 # allocates a tiny bit
  ss_targpdiffs = sum(targpdiffs.^2.)
  avg_tdiff = ss_targpdiffs / length(targpdiffs)

  # part 2 - measure of change in ratio
  ratiodiffs = ratio .- 1.0
  ss_ratiodiffs = sum(ratiodiffs.^2.)
  avg_rdiff = ss_ratiodiffs / length(ratiodiffs)

  # combine the two measures and (maybe later) take a root
  # objval = (ss_targdiffs / length(targdiffs))*(1. - whweight) +
  #         (ss_whdiffs / length(whdiffs))*whweight
  # objval = objval^(1. / pow)  
  # objval = avg_tdiff*(1 - rweight) + avg_rdiff*rweight
  objval = avg_tdiff*(1 - rweight) + avg_rdiff*rweight

  # list extra variables on the return so that they are available to the callback function
  # all returned variables must be arguments of the callback function
  return objval, targpdiffs, ratiodiffs # values to be used in callback function must be returned here
end


# %% minimum error solve functions

function rwminerr_spg(wh, xmat, rwtargets;
  ratio0=ones(length(wh)),
  lb=0.1,
  ub=10.0,
  rweight=0.5,
  maxiters=1000)

  # f = (ratio) -> objfn_reweight(ratio, wh, xmat, rwtargets, rweight=0.0)
  # g = (ratio) -> ReverseDiff.gradient(f, ratio)
  # g2(g2, ratio) -> ReverseDiff.gradient!(g2, f2, ratio)


  # opt = spgbox(f, (g,x) -> ReverseDiff.gradient!(g,f,x), x=x, lower=lower, upper=upper, eps=1e-16, nitmax=maxiters, nfevalmax=20000, m=10, iprint=0)

  wh2 = wh # why did i do this - was it needed in f? check
  f = (ratio) -> objfn_reweight(ratio, wh2, xmat, rwtargets, rweight=rweight)
  # g2 = (ratio) -> ReverseDiff.gradient(f2, ratio)

  lower = fill(lb, length(ratio0)) # can't use scalar
  upper = fill(ub, length(ratio0))

  x = ratio0
  # println("x: $x")
  # println("wh: $wh")
  # println("xmat: $xmat")
  # println("rwtargets: $rwtargets")
  # println("rweight: $rweight")
  # println("f(ratio0): ", f(ratio0))

  opt = spgbox(f, (g,x) -> ReverseDiff.gradient!(g,f,x), x, lower=lower, upper=upper, eps=1e-16, nitmax=10000, nfevalmax=20000, m=10, iprint=0)
  return opt
end

function rwminerr_nlopt(wh, xmat, rwtargets;
  ratio0=ones(length(wh)),
  algo="LD_LBFGS",
  lb=0.1,
  ub=10.0,
  rweight=0.5,
  scaling=false,
  maxiters=1000)

  # convert the string nloptfname into a proper symbol
  # NLOPT algorithms that (1) find local optima (L), (2) use derivatives (D) -- i.e., LD -- and
  # can handle box constraints
  allowable_algorithms = ["LD_CCSAQ", "LD_LBFGS", "LD_MMA", "LD_VAR1", "LD_VAR2", 
  "LD_TNEWTON", "LD_TNEWTON_RESTART", "LD_TNEWTON_PRECOND_RESTART", "LD_TNEWTON_PRECOND"]

  # LD_TNEWTON seems best when we inlude targets and ratios in the objective function
  # LD_CCSAQ best on many real-world problems
  # LD_LBFGS good alternative
  # LD_MMA also good
  # LD_TNEWTON_PRECOND seems to fail

  # for future use, here is how to pass NLOPT options:
  #   algorithm=:(LD_LBFGS(M=20))  

   if !(algo in allowable_algorithms)
      throw(ArgumentError("ERROR: algo was $algo -- its value must be in: $allowable_algorithms"))
   end

   fsym = Symbol(algo)
   algorithm = Expr(:call, fsym) # a proper symbol for the function name

   if scaling 
     xmat, rwtargets = rwscale(xmat, rwtargets)
   end

   p = 1.0
   fp = (ratio, p) -> objfn_reweight(ratio, wh, xmat, rwtargets, rweight=rweight)
   fpof = Optimization.OptimizationFunction{true}(fp, Optimization.AutoZygote())
   fprob = Optimization.OptimizationProblem(fpof, ratio0, lb=lb, ub=ub) # rerun this line when ratio0 changes

  # %% setup preallocations
   # p = 1.0
  # p_mshares = Array{Float64,2}(undef, prob.h, prob.s)
  # A = zeros(prob.h, prob.s)
  # p_whs = Array{Float64,2}(undef, prob.h, prob.s)
  # p_calctargets = Array{Float64,2}(undef, prob.s, prob.k)
  # p_pdiffs = Array{Float64,2}(undef, prob.s, prob.k)
  # p_whpdiffs = Array{Float64,1}(undef, prob.h)

  opt = Optimization.solve(fprob, NLopt.eval(algorithm), maxiters=maxiters, reltol=1e-16, callback=cb_test) # callback=cb_rwminerr , callback=cb_test

  return opt
end


function rwmconstrain_ipopt(wh, xmat, rwtargets;
  ratio0=ones(length(wh)),
  lb=0.1,
  ub=10.0,
  constol=0.01,
  scaling=false,
  maxiters=1000)

  lvar = fill(lb, length(wh))
  uvar = fill(ub, length(wh))
  lcon = rwtargets .- abs.(rwtargets)*constol
  ucon = rwtargets .+ abs.(rwtargets)*constol

  # A = xmat .* wh
  mod = modcon(xmat .* wh, rwtargets; lvar=lvar, uvar=uvar, lcon=lcon, ucon=ucon) # fill the specialized structure used by NLPModels

  opt = ipopt(mod, print_level=5, hessian_constant="yes", jac_c_constant="yes", jac_d_constant="yes", linear_solver="mumps", mumps_mem_percent=50)


  # safe way to run ma77 - avoids crash -- this is important
  # hsllib = "/usr/local/lib/lib/x86_64-linux-gnu/libcoinhsl.so"
  # tempdir will store the temp files that ma77 creates; we'll delete it after the run because ma77 does not always clean up
  # tempdir_path = Base.mktempdir()
  # cd(tempdir_path)
  # res2 = ipopt(mod, print_level=5, hessian_constant="yes", jac_c_constant="yes", jac_d_constant="yes", hsllib=hsllib, linear_solver="ma77")
  # cd("..")  # Change back to the parent directory (original directory)
  # rm(tempdir_path; recursive=true, force=true)

  return opt
end

