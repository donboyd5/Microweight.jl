
# module BruteForce

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
  rweight=0.5,
  pow=2.0,
  targstop=true, whstop=true,
  display_progress=true)

  # part 1 get measure of difference from targets
  rwtargets_calc = xmat' * (ratio .* wh)
  targpdiffs = (rwtargets_calc .- rwtargets) ./ rwtargets # ./ 1e6 # allocates a tiny bit
  ss_targpdiffs = sum(targpdiffs.^2.)
  avg_tdiff = ss_targpdiffs / length(targpdiffs)

  # part 2 - measure of change in ratio
  ratiodiffs = ratio .- 1.0
  ss_ratiodiffs = sum(ratiodiffs.^6.)
  avg_rdiff = ss_ratiodiffs / length(ratiodiffs)

  # combine the two measures and (maybe later) take a root
  # objval = (ss_targdiffs / length(targdiffs))*(1. - whweight) +
  #         (ss_whdiffs / length(whdiffs))*whweight
  # objval = objval^(1. / pow)  
  # objval = avg_tdiff*(1 - rweight) + avg_rdiff*rweight
  objval = avg_tdiff*(1 - rweight) + avg_rdiff*rweight

  # list extra variables on the return so that they are available to the callback function
  return objval # , targdiffs, whdiffs, targstop, whstop
end


# %% solve function

function rwsolve(wh, xmat, rwtargets;
  ratio0=ones(length(wh)),
  algo="LD_TNEWTON",
  lb=0.1,
  ub=10.0,
  rweight=0.5,
  scaling=false,
  maxit=1000)

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
        throw(ArgumentError("ERROR: The value of algo must be in: $allowable_algorithms"))
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

  opt = Optimization.solve(fprob, NLopt.eval(algorithm), maxiters=maxit, reltol=1e-16)

  return opt
end

