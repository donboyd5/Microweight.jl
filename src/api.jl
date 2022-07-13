
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
            whweight=nothing,
            pow=nothing,
            targstop=.01,
            whstop=.01,
            kwargs...)
    # allowable methods:
    #   lm_lsqfit, lm_minpack
    println("Solving problem...")

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

    if approach == :poisson
        if method == :cg_optim
            # poisson_cgoptim(prob, result; maxiter=maxiter, objscale=objscale, interval, kwargs...)
            poisson_cgoptim(prob, result; maxiter=100, objscale, interval=interval, targstop=targstop, whstop=whstop, kwargs...)
        elseif method == :cg_optim2
            poisson_cgoptim2(prob, result; maxiter=maxiter, objscale=objscale, interval, kwargs...)
        elseif method == :lm_lsoptim
            poisson_lsoptim(prob, result; maxiter=maxiter, objscale=objscale, interval, kwargs...)
        elseif method == :lm_lsqfit
            poisson_lsqlm(prob, result; maxiter=maxiter, objscale=objscale, kwargs...)
        elseif method == :lm_minpack
            poisson_minpack(prob, result; maxiter=maxiter, objscale=objscale, interval, kwargs...)
        # elseif method == :lm_mads
        #     poisson_mads(prob, beta0, result; maxiter=maxiter, objscale=objscale, kwargs...)
        elseif method == :newttr_nlsolve
            poisson_newttrust(prob, result; maxiter=maxiter, objscale=objscale, interval, kwargs...)
        elseif method == :krylov
            poisson_krylov(prob, result; maxiter=maxiter, objscale=objscale, interval, kwargs...)
        else
            error("Unknown poisson method!")
            return;
        end
    elseif approach == :direct
        nlopt_methods = (:ccsaq, :lbfgs_nlopt, :mma, :newton, :newtonrs, :var1, :var2)
        optim_methods = (:cg, :gd, :lbfgs_optim)
        # println("ok methods: $okmethod")

        if method in optim_methods
            direct_optim(prob, result, pow=pow, maxiter=maxiter, interval=interval,
                whweight=whweight, targstop=targstop, whstop=whstop; kwargs...)
        elseif method in nlopt_methods
            # kwargs are those that should be passed through to NLopt from Optimization
            direct_nlopt(prob, result, pow=pow, maxiter=maxiter, interval=interval,
                whweight=whweight, targstop=targstop, whstop=whstop; kwargs...)
        else
            error("Unknown direct method!")
            return;
        end
    else
        error("Unknown approach!")
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

    return result
end
