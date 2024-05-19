
function geosolve(prob;
            approach=nothing,
            method=nothing,
            beta0=nothing,
            shares0=nothing,
            maxiter=1000,
            objscale=1.0,
            scaling=false,
            scaling_target_goal=1000.0,
            print_interval=1,
            whweight=0.5,
            pow=8,
            targstop=.01,
            whstop=.01,
            kwargs...)
    # allowable methods:
    #   lm_lsqfit, lm_minpack
    println("Solving geoweighting problem...\n")

    # define defaults
    if isnothing(approach) approach=:poisson end

    if approach==:poisson
        if isnothing(method) method=:lm_lsqfit end
        if isnothing(beta0) beta0 = zeros(length(prob.geotargets)) end
    elseif approach==:direct
        if isnothing(method) method=:ccsaq end
        if isnothing(shares0) shares0=fill(1. / prob.s, prob.h * prob.s) end
    else
        return "ERROR: approach must be :poisson or :direct"
    end
    println("approach: ", approach)
    println("method: ", method)

    # initialize result
    prob = scale_prob(prob, scaling=scaling, scaling_target_goal=scaling_target_goal)
    result = Result(approach=approach, method=method, problem=prob, beta0=beta0, shares0=shares0)

    # globals accessed within the display_progress function
    global tstart = time()
    global fcalls = 0  # global within this module
    global bestobjval = Inf
    global nshown = 0
    global iter_calc = 0
    global plevel = .99
    global interval = print_interval

    # NOTES:
    #   poisson method, we need lm or rootfinding -- scalar optimizers don't work well
    #   direct method: ccsaq works best

    if approach == :poisson
        minpack_methods = (:hybr_minpack, :lm_minpack)
        nlopt_methods = (:ccsaq, :lbfgs_nlopt, :mma, :newton, :newtonrs, :var1, :var2)
        # only ccsaq and mma work well, others do not progress (why??)
        # nlsolve_methods = (:anderson, :broyden, :newton_nlsolve, :trust_nlsolve)
        # do not use:
        #   anderson - generates LAPACK exception
        #   broyden - does not reach good results
        nlsolve_methods = (:newton_nlsolve, :trust_nlsolve) # only allow these
        optim_methods = (:cg, :gd, :lbfgs_optim, :krylov) # , :newton_optim
        optimisers_methods = (:adam,  :descent, :momentum, :nesterov)

        if method == :lm_lsoptim   # objective function returns a vector
            # LsqFit.levenberg_marquardt does not have stopping criteria or allow callbacks
            poisson_lsoptim(prob, result; maxiter=maxiter, objscale=objscale, kwargs...)
        elseif method == :lm_lsqfit   # objective function returns a vector
            poisson_lsqlm(prob, result; maxiter=maxiter, objscale=objscale, kwargs...)
        elseif method in minpack_methods # objective function returns a vector
            poisson_minpack_fsolve(prob, result; maxiter=maxiter, objscale=objscale, kwargs...)
        elseif method in nlsolve_methods # objective function returns a vector
            poisson_nlsolve(prob, result; maxiter=maxiter, objscale=objscale, kwargs...)
        # elseif method == :krylov # objective function returns a vector
        #     poisson_krylov(prob, result; maxiter=maxiter, objscale=objscale, kwargs...)

        elseif method in nlopt_methods # objective function returns a scalar
            poisson_optz_nlopt(prob, result; maxiter=maxiter, pow=pow, targstop=targstop, whstop=whstop, objscale=objscale, kwargs...)
        elseif method in optim_methods # objective function returns a scalar, thus I can modify with powers
            poisson_optz_optim(prob, result, maxiter=maxiter, objscale=objscale, pow=pow, targstop=targstop, whstop=whstop; kwargs...)
        elseif method in optimisers_methods # objective function returns a scalar, thus I can modify with powers
            poisson_optz_optimisers(prob, result, maxiter=maxiter, objscale=objscale, pow=pow, targstop=targstop, whstop=whstop; kwargs...)

        else
            error("Unknown poisson geoweighting method!")
            return;
        end

    elseif approach == :direct
        nlopt_methods = (:ccsaq, :lbfgs_nlopt, :mma, :newton, :newtonrs, :var1, :var2)
        optim_methods = (:cg, :gd, :lbfgs_optim)
        optimisers_methods = (:adam, :nesterov, :descent, :momentum)

        if method in optim_methods
            direct_optz_optim(prob, result, pow=pow, maxiter=maxiter,
                whweight=whweight, targstop=targstop, whstop=whstop; kwargs...)
        elseif method == :krylov
            # kwargs are those that should be passed through to NLopt from Optimization
            direct_optz_optim_krylov(prob, result, pow=pow, maxiter=maxiter,
                whweight=whweight, targstop=targstop, whstop=whstop; kwargs...)
        elseif method in nlopt_methods
            # kwargs are those that should be passed through to NLopt from Optimization
            direct_optz_nlopt(prob, result, pow=pow, maxiter=maxiter,
                whweight=whweight, targstop=targstop, whstop=whstop; kwargs...)
        elseif method in optimisers_methods # objective function returns a scalar, thus I can modify with powers
            direct_optz_optimisers(prob, result, pow=pow, maxiter=maxiter,
                whweight=whweight, targstop=targstop, whstop=whstop; kwargs...)
        else
            error("Unknown direct geoweighting method!")
            return;
        end
    else
        error("Unknown geoweighting approach!")
    end

    tend = time()
    result.eseconds = tend - tstart

    if result.success || (result.iterations >= maxiter) || (result.iterations == -9)
        p = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]

        if approach == :poisson
            result.whs = geo_weights(result.beta, prob.wh, prob.xmat_scaled, size(prob.geotargets))
        elseif approach == :direct
            result.whs = fwhs(result.shares, prob.wh)
        end

        result.wh_calc = sum(result.whs, dims=2)
        result.wh_pdiffs =  (result.wh_calc - prob.wh) ./ prob.wh * 100.
        result.wh_pdqtiles = Statistics.quantile(vec(result.wh_pdiffs), p)
        result.geotargets_calc = geo_targets(result.whs, prob.xmat)
        result.sspd = sspd(result.geotargets_calc, prob.geotargets)
        result.targ_pdiffs = targ_pdiffs(result.geotargets_calc, prob.geotargets)
        result.targ_pdqtiles = Statistics.quantile(vec(result.targ_pdiffs), p)
    end
    println()

    return result
end


function rwsolve(prob;
    approach=nothing, # :minerr, :constrain
    method=nothing, # depends on approach
    lb=.01,
    ub=10.,
    rweight=0.5,
    constol=.01,
    maxiters=1000,
    objscale=1.0,
    scaling=false,
    scaling_target_goal=1000.0,
    print_interval=1,
    targstop=.01,
    kwargs...)

    println("\nSolving reweighting problem...\n")

    # globals accessed within the callback function
    global tstart = time()
    global fcalls = 0  # global within this module
    global bestobjval = Inf
    global nshown = 0
    global iter_calc = 0
    global plevel = .95
    global interval = print_interval

    # check inputs, add defaults as needed
    function print_prob()
        println("households: ", prob.h)
        println("targets: ", prob.k)
        println("approach: ", approach)
        println("method used: ", method)
        println("scaling: ", scaling)
        println("lb: $lb")
        println("lb: $ub")
        println("rweight $rweight")
        println("constraints tolerance: $constol")
        println("maxiters: $maxiters")
    end

    # check method and decide on solver
    if approach==:minerr
        if isnothing(method) 
            method="spg"  # "LD_CCSAQ" # LBFGS seems best when ratio error is most important, CCSAQ when target error is most important
            println("method nothing changed to :minerr default: $method")
        end
    elseif approach==:constrain
        if isnothing(method) 
            method = "ipopt" 
            println("method nothing changed to :constrain default: $method")
        end
    end

    # initialize result
    # prob = scale_prob(prob, scaling=scaling, scaling_target_goal=scaling_target_goal)
    result = ReweightResult(approach=approach, method=method, rwtargets=prob.rwtargets, wh=prob.wh, xmat=prob.xmat, h=length(prob.wh), k=size(prob.xmat)[2])

    if approach==:minerr        
        nlopt_algorithms = ["LD_CCSAQ", "LD_LBFGS", "LD_MMA", "LD_VAR1", "LD_VAR2", "LD_TNEWTON", "LD_TNEWTON_RESTART", "LD_TNEWTON_PRECOND_RESTART", "LD_TNEWTON_PRECOND"]
        optim_algorithms = ["LBFGS", "KrylovTrustRegion"]
        if method in nlopt_algorithms
            print_prob()
            println("\nBeginning solve...")
            opt = rwminerr_nlopt(prob.wh, prob.xmat, prob.rwtargets, method=method, lb=lb, ub=ub, rweight=rweight, maxiters=maxiters, targstop=targstop, scaling=scaling; kwargs...)
            println(fieldnames(typeof(opt)))
            # success, :iterations, :eseconds, :objval, :sspd, :rwtargets, :rwtargets_calc, :targ_pdiffs, :targ_pdqtiles, :solver_result, :h, :k, :wh, :xmat, :scaling)
            # (:u, :cache, :alg, :objective, :retcode, :original, :solve_time, :stats)
            result.success = opt.retcode
            result.objval = opt.objective
            result.x = opt.u
        elseif method in optim_algorithms
            print_prob()
            println("\nBeginning solve...")
            opt = rwminerr_optim(prob.wh, prob.xmat, prob.rwtargets, method=method, lb=lb, ub=ub, rweight=rweight, maxiters=maxiters, targstop=targstop, scaling=scaling)
            println(fieldnames(typeof(opt)))
            # success, :iterations, :eseconds, :objval, :sspd, :rwtargets, :rwtargets_calc, :targ_pdiffs, :targ_pdqtiles, :solver_result, :h, :k, :wh, :xmat, :scaling)
            # (:u, :cache, :alg, :objective, :retcode, :original, :solve_time, :stats)
            result.success = opt.retcode
            result.objval = opt.objective
            result.x =opt.u 
        elseif method=="spg"
            print_prob()
            opt = rwminerr_spg(prob.wh, prob.xmat, prob.rwtargets, lb=lb, ub=ub, rweight=rweight, maxiters=maxiters, targstop=targstop, scaling=scaling)
            println(fieldnames(typeof(opt)))
            # success, :iterations, :eseconds, :objval, :rwtargets, :rwtargets_calc, :targ_pdiffs, :targ_pdqtiles, :solver_result, :h, :k, :wh, :xmat, :scaling)
            # (:x, :f, :gnorm, :nit, :nfeval, :ierr, :return_from_callback)
            # result.success = opt.retcode
            result.objval = opt.f
            result.x = opt.x
            result.iterations = opt.nit
        else
            println("unknown method $method")
            return "not attempted"
        end
        
    elseif approach==:constrain        
        # tulip here
        if method == "ipopt"
            print_prob()
            opt = rwmconstrain_ipopt(prob.wh, prob.xmat, prob.rwtargets; lb=lb, ub=ub, constol=constol, maxiters=maxiters, targstop=targstop, scaling=scaling)
            println(fieldnames(typeof(opt)))
            # success, :iterations, :eseconds, :objval, :rwtargets, :rwtargets_calc, :targ_pdiffs, :targ_pdqtiles, :solver_result, :h, :k, :wh, :xmat, :scaling)
            # status_reliable, :status, :solution_reliable, :solution, :objective_reliable, :objective, :dual_residual_reliable, :dual_feas, :primal_residual_reliable, 
            #   :primal_feas, :multipliers_reliable, :multipliers, :bounds_multipliers_reliable, :multipliers_L, :multipliers_U, :iter_reliable, :iter, :time_reliable,
            #   :elapsed_time, :solver_specific_reliable, :solver_specific
            result.objval = opt.objective
            result.x = opt.solution
            result.success = opt.status
            result.iterations = opt.iter
        elseif method == "tulip"
            print_prob()
            opt = rwmconstrain_tulip(prob.wh, prob.xmat, prob.rwtargets; lb=lb, ub=ub, constol=constol, maxiters=maxiters, scaling=scaling)
            result.objval = opt.objval
            result.x = opt.x
            result.iterations = opt.iterations
            return opt
        else
            return "unknown method"
        end
    else # some other approach
        return "ERROR: approach must be :minerr or :constrain"
    end

    tend = time()
    result.eseconds = tend - tstart
    println("neseconds: $(result.eseconds)")

    result.solver_result = opt

    return result

    # DJB have not yet updated this function
    # initialize result
    prob = scale_prob(prob, scaling=scaling, scaling_target_goal=scaling_target_goal)
    result = Result(approach=approach, method=method, problem=prob, beta0=beta0, shares0=shares0)

    # globals accessed within the display_reweight_progress function (TO BE WRITTEN)
    global tstart = time()
    global fcalls = 0  # global within this module
    global bestobjval = Inf
    global nshown = 0
    global iter_calc = 0
    global plevel = .99
    global interval = print_interval


    nlopt_methods = (:ccsaq, :lbfgs_nlopt, :mma, :newton, :newtonrs, :var1, :var2)
    optim_methods = (:cg, :gd, :lbfgs_optim)
    optimisers_methods = (:adam, :nesterov, :descent, :momentum)

    if method in optim_methods
        error("Not yet implemented!")
        reweight_optz_optim(prob, result, pow=pow, maxiter=maxiter,
            whweight=whweight, targstop=targstop, whstop=whstop; kwargs...)
    elseif method == :krylov
        error("Not yet implemented!")
        # kwargs are those that should be passed through to NLopt from Optimization
        reweight_optz_optim_krylov(prob, result, pow=pow, maxiter=maxiter,
            whweight=whweight, targstop=targstop, whstop=whstop; kwargs...)
    elseif method in nlopt_methods
        error("Not yet implemented!")
        # kwargs are those that should be passed through to NLopt from Optimization
        reweight_optz_nlopt(prob, result, pow=pow, maxiter=maxiter,
            whweight=whweight, targstop=targstop, whstop=whstop; kwargs...)
    elseif method in optimisers_methods # objective function returns a scalar, thus I can modify with powers
        error("Not yet implemented!")
        reweight_optz_optimisers(prob, result, pow=pow, maxiter=maxiter,
            whweight=whweight, targstop=targstop, whstop=whstop; kwargs...)
    else
        error("Unknown reweighting method!")
        return;
    end

    return result
end


