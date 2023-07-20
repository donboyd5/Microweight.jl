
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


function rwsolve(wh, xmat, rwtargets;
  ratio0=ones(length(wh)),
  algo="LD_CCSAQ",
  scaling=false)

  # convert the string nloptfname into a proper symbol
  fsym = Symbol(algo)
  algorithm = Expr(:call, fsym) # a proper symbol for the function name

  if scaling 
    xmat, rwtargets = rwscale(xmat, rwtargets)
  end

  p = 1.0
  fp = (ratio, p) -> objfn_reweight(ratio, wh, xmat, rwtargets)
  fpof = Optimization.OptimizationFunction{true}(fp, Optimization.AutoZygote())
  fprob = Optimization.OptimizationProblem(fpof, ratio0, lb=0.1, ub=10.0) # rerun this line when ratio0 changes

  opt = Optimization.solve(fprob, NLopt.eval(algorithm), maxiters=1000, reltol=1e-16)

  return opt
end


# %% opt functions

function objfn_reweight(
  ratio, wh, xmat, rwtargets;
  whweight=0.5,
  pow=2.0,
  scaling=true,
  targstop=true, whstop=true,
  display_progress=true)

  # part 1 get measure of difference from targets
  rwtargets_calc = xmat' * (ratio .* wh)
  targpdiffs = (rwtargets_calc .- rwtargets) ./ rwtargets # ./ 1e6 # allocates a tiny bit
  ss_targpdiffs = sum(targpdiffs.^pow)
  objval = ss_targpdiffs

  # part 2 - measure of change in weight
  # whdiffs = wh ./ wh .- 1.0
  # ss_whdiffs = mean(whdiffs .^pow)

  # combine the two measures and take a root
  # objval = (ss_targdiffs / length(targdiffs))*(1. - whweight) +
  #         (ss_whdiffs / length(whdiffs))*whweight
  # objval = objval^(1. / pow)

  # list extra variables on the return so that they are available to the callback function
  return objval # , targdiffs, whdiffs, targstop, whstop
end

