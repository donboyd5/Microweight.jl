
# module BruteForce

# https://docs.sciml.ai/Optimization/stable/API/optimization_function/
# AutoForwardDiff(): The fastest choice for small optimizations
# AutoReverseDiff(compile=false): A fast choice for large scalar optimizations
# AutoTracker(): Like ReverseDiff but GPU-compatible
# AutoZygote(): The fastest choice for non-mutating array-based (BLAS) functions
# AutoFiniteDiff(): Finite differencing, not optimal but always applicable
# AutoModelingToolkit(): The fastest choice for large scalar optimizations


# %% utility functions

function rwscaleAb(A, b)
  # my A has m=# vars, n=#constraints (transpose of the usual A)
  # try to scale A so that largest coefficient (in absolute value) of each COLUMN is about 1
  # or perhaps have them in 1e-4, 1e4
  # scale b accordingly

  println("scaling!")
  maxcoeff = vec(maximum(abs.(A), dims=1)) # largest coefficient in each column
  scale = 1e0 ./ maxcoeff
  A = A .* scale'
  b = b .* scale

  return A, b
end


function rwscale(wh, xmat, rwtargets)
  println("scaling!")
  maxcoeff = vec(maximum(abs.(xmat .* wh), dims=1)) # largest coefficient in each column
  scale = 1e0 ./ maxcoeff
  xmat = xmat .* scale'
  rwtargets = rwtargets .* scale
  return xmat, rwtargets
end


function rwscale_old(xmat, rwtargets)
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
  ratio, wh2, xmat2, rwtargets2;
  rweight=0.1, # relative importance of minimizing ratio error rather than target error
  method="LD_CCSAQ",
  targstop2=.01,
  display_progress=true)

  # define variables needed for the spg callback - these must be available to the closure function cb_spg
  global wh = wh2
  global xmat = xmat2
  global rwtargets = rwtargets2
  global targstop = targstop2

  # println("targstop: $targstop")

  # part 1 get measure of difference from targets
  rwtargets_calc = xmat' * (ratio .* wh)
  targpdiffs = (rwtargets_calc .- rwtargets) ./ rwtargets # ./ 1e6 # allocates a tiny bit
  targ_sse = sum(targpdiffs.^2.)
  targ_rmse = targ_sse / length(targpdiffs)

  # part 2 - measure of change in ratio
  ratiodiffs = ratio .- 1.0
  ratio_sse = sum(ratiodiffs.^2.)
  ratio_rmse = ratio_sse / length(ratiodiffs)

  # combine the two measures and (maybe later) take a root
  objval = targ_rmse*(1 - rweight) + ratio_rmse*rweight

  # list extra variables on the return so that they are available to the callback function
  # all returned variables must be arguments of the callback function
  # if method=="LD_LBFGS"
  #   # https://nlopt.readthedocs.io/en/latest/NLopt_Reference/#forced-termination
  #   if targ_rmse < .01 nlopt_result nlopt_force_stop(nlopt_opt opt) end
  # end

  # cb_spg(R::SPGBox.SPGBoxResult) = cb_spg(R, wh, xmat, rwtargets)

  if method != "spg"
    return objval, targ_rmse, targpdiffs, ratio_rmse, ratiodiffs, targstop # values to be used in an Optimization.jl callback function must be returned here
  elseif method == "spg"
    return objval
  end
end


# %% minimum error solve functions

function rwminerr_spg(wh, xmat, rwtargets;
  ratio0=ones(length(wh)),
  method="spg",
  lb=0.1,
  ub=10.0,
  rweight=0.5,
  maxiters=1000,
  targstop=.01,
  scaling=false)

  # short term approach to scaling: 
  if scaling
    xmat, rwtargets = rwscale(wh, xmat, rwtargets)
  end

  f = (ratio) -> objfn_reweight(ratio, wh, xmat, rwtargets, rweight=rweight, method=method, targstop2=targstop) # note it is targstop2

  lower = fill(lb, length(ratio0)) # can't use scalar
  upper = fill(ub, length(ratio0))

  x = ratio0  
  # println("targstop: $targstop")

  opt = spgbox(f, (g,x) -> ReverseDiff.gradient!(g,f,x), x, lower=lower, upper=upper, eps=1e-16, nitmax=10000, nfevalmax=20000, m=10, iprint=0, callback=cb_spg)
  return opt
end


function rwminerr_spg_save(wh, xmat, rwtargets;
  ratio0=ones(length(wh)),
  method="spg",
  lb=0.1,
  ub=10.0,
  rweight=0.5,
  maxiters=1000,
  targstop=.01)

  f = (ratio) -> objfn_reweight(ratio, wh, xmat, rwtargets, rweight=rweight, method=method, targstop2=targstop) # note it is targstop2

  lower = fill(lb, length(ratio0)) # can't use scalar
  upper = fill(ub, length(ratio0))

  x = ratio0  
  # println("targstop: $targstop")

  opt = spgbox(f, (g,x) -> ReverseDiff.gradient!(g,f,x), x, lower=lower, upper=upper, eps=1e-16, nitmax=10000, nfevalmax=20000, m=10, iprint=0, callback=cb_spg)
  return opt
end


function rwminerr_nlopt(wh, xmat, rwtargets;
  method="LD_CCSAQ",
  ratio0=ones(length(wh)),
  lb=0.1,
  ub=10.0,
  rweight=0.5,
  targstop=.01,
  scaling=false,
  maxiters=1000,
  kwargs...) # default stopval=1e-6

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

   if !(method in allowable_algorithms)
      throw(ArgumentError("ERROR: method was $method -- its value must be in: $allowable_algorithms"))
   end

   fsym = Symbol(method)
   algorithm = Expr(:call, fsym) # a proper symbol for the function name

  if scaling
    xmat, rwtargets = rwscale(wh, xmat, rwtargets)
  end

   fp = (ratio, p) -> objfn_reweight(ratio, wh, xmat, rwtargets, rweight=rweight, method=method, targstop2=targstop)
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

  # djb todo: pass stopval earlier!!
  # stopval = 1e-12 * size(xmat)[1] * size(xmat)[2] * .6
  stopval = get(kwargs, :stopval, nothing)
  if stopval == nothing
    # ntargs = size(xmat)[2]
    # nratios = size(xmat)[1]
    avgtargerr = .005
    # avgratioerr = 0.05
    # stopval = (avgtargerr ^2.) * (1. - rweight) + (avgratioerr^2.) * rweight
    stopval = (avgtargerr ^2.) * .25
    println("stopval: ", stopval)

    opt = Optimization.solve(fprob, NLopt.eval(algorithm), maxiters=maxiters, reltol=1e-16, 
      callback=cb_rwminerr_nlopt, stopval=stopval, kwargs...) # , callback=cb_rwminerr cb_test
  else
    println("stopval: ", stopval)
    opt = Optimization.solve(fprob, NLopt.eval(algorithm), maxiters=maxiters, reltol=1e-16, callback=cb_rwminerr_nlopt; kwargs...) # , callback=cb_rwminerr cb_test
  end

  return opt
end

# optim_algorithms = ["LBFGS", "KrylovTrustRegion"]

function rwminerr_optim(wh, xmat, rwtargets;
  method="LBFGS",
  ratio0=ones(length(wh)),
  lb=0.1,
  ub=10.0,
  rweight=0.5,
  targstop=.01,
  scaling=false,
  maxiters=200)

  # convert the string method into a proper symbol
  # allowable_algorithms = ["LBFGS", "KrylovTrustRegion"]
  allowable_algorithms = ["LBFGS"]

  # for future use, here is how to pass NLOPT options:
  #   algorithm=:(LD_LBFGS(M=20))  

   if !(method in allowable_algorithms)
      throw(ArgumentError("ERROR: method was $method -- its value must be in: $allowable_algorithms"))
   end

   fsym = Symbol(method)
   algorithm = Expr(:call, fsym) # a proper symbol for the function name

  if scaling
    xmat, rwtargets = rwscale(wh, xmat, rwtargets)
  end

  fp = (ratio, p) -> objfn_reweight(ratio, wh, xmat, rwtargets, rweight=rweight, method=method, targstop2=targstop)
  # defaults: m=10, alphaguess = LineSearches.InitialStatic(), linesearch = LineSearches.HagerZhang()
  # algo_hz = Newton(alphaguess = InitialHagerZhang(α0=1.0), linesearch = HagerZhang())
  fpof = Optimization.OptimizationFunction{true}(fp, Optimization.AutoZygote())
  fprob = Optimization.OptimizationProblem(fpof, ratio0, lb=lb, ub=ub) # rerun this line when ratio0 changes

  # alphaguess = InitialHagerZhang(α0=1.0) slightly better max
  # linesearch = LineSearches.StrongWolfe(), MoreThuente fast + BAD;  Static BackTracking bad
  # opt = Optimization.solve(fprob, Optim.LBFGS(m=10, alphaguess = InitialQuadratic(), linesearch = LineSearches.HagerZhang()),
  #  maxiters=maxiters, reltol=1e-16, callback=cb_rwminerr) # , callback=cb_rwminerr cb_test
  opt = Optimization.solve(fprob, Optim.eval(algorithm), maxiters=maxiters, reltol=1e-16, callback=cb_rwminerr) # BASE CASE
  # opt = Optimization.solve(fprob, Fminbox(Optim.KrylovTrustRegion()), maxiters=maxiters, reltol=1e-16, callback=cb_rwminerr)

  return opt
end


function rwmconstrain_ipopt(wh, xmat, rwtargets;
  ratio0=ones(length(wh)),
  lb=0.1,
  ub=10.0,
  constol=0.01,
  targstop=nothing,
  scaling=false,
  maxiters=1000)

  if scaling
    xmat, rwtargets = rwscale(wh, xmat, rwtargets)
  end

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



function rwmconstrain_tulip(wh, xmat, rwtargets;
  lb=0.1,
  ub=10.0,
  constol=0.01,
  scaling=false,
  maxiters=100)

  N = length(wh)
  A = xmat .* wh
  initval = vec(sum(A, dims=1))
  b = rwtargets .- initval

  if scaling
    A, b = rwscaleAb(A, b)
  end

  lcon = b .- abs.(b)*constol
  ucon = b .+ abs.(b)*constol

  model = Model(Tulip.Optimizer)
  set_optimizer_attribute(model, "OutputLevel", 1)  # 0=disable output (default), 1=show iterations
  set_optimizer_attribute(model, "IPM_IterationsLimit", maxiters)  # default 100 seems to be enough

  # we split the amount by which x is above or below 1 into two parts, r and s
  # this allows us to minimize the sum of the absolute values of the differences in ratios from 1
  @variable(model, 0.0 <= r[1:N] <= lb + ub) # r is the part of the ration that is above 1 e.g., 1 + 0.40
  @variable(model, 0.0 <= s[1:N] <= lb + ub) # s is the part below 1

  @objective(model, Min, sum(r + s)); 

  # constraints
  # initval = vec(sum(A, dims=1))
  @constraint(model, lcon .<= (A' * r) - (A' * s) .<= ucon); 
  @constraint(model, lb .<= 1.0 .+ (r - s) .<= ub); # all dots needed for broadcasting

  JuMP.optimize!(model);

  objval = objective_value(model)
  iterations = barrier_iterations(model)
  x = 1.0 .+ (JuMP.value.(r) - JuMP.value.(s)) 
  
  opt = (objval=objval, iterations=iterations, x=x)

  return opt
end
