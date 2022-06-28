
function geosolve(prob; approach=:poisson, method=:lm_lsqfit, beta0=zeros(length(prob.geotargets)),
    maxiter=100, objscale=1.0, scaling=false, scaling_target_goal=1000.0,
    interval=1,
    kwargs...)
    # allowable methods:
    #   lm_lsqfit, lm_minpack
    println("Solving problem...")
    result = Result(method=method)

    global tstart = time()

    prob = scale_prob(prob, scaling=scaling, scaling_target_goal=scaling_target_goal)

    result.problem = prob

    global fcalls = 0  # global within this module

    if approach == :poisson
        if method == :cg_optim
            cg_optim(prob, beta0, result; maxiter=maxiter, objscale=objscale, interval, kwargs...)
        elseif method == :lm_lsoptim
            lsoptim(prob, beta0, result; maxiter=maxiter, objscale=objscale, interval, kwargs...)
        elseif method == :lm_lsqfit
            lsqlm(prob, beta0, result; maxiter=maxiter, objscale=objscale, interval, kwargs...)
        elseif method == :lm_minpack
            minpack(prob, beta0, result; maxiter=maxiter, objscale=objscale, interval, kwargs...)
        # elseif method == :lm_mads
        #     mads(prob, beta0, result; maxiter=maxiter, objscale=objscale, kwargs...)
        elseif method == :newttr_nlsolve
            newttr_nlsolve(prob, beta0, result; maxiter=maxiter, objscale=objscale, interval, kwargs...)
        elseif method == :krylov
            algo_optz(prob, beta0, result; maxiter=maxiter, objscale=objscale, interval, kwargs...)
        else
            error("Unknown poisson method!")
            return;
        end
    elseif approach == :direct
        if method==:direct_cg
            direct_cg(prob, result)
        else
            error("Unknown direct method!")
            return;
        end
    else
        error("Unknown approach!")
    end

    tend = time()
    result.eseconds = tend - tstart

    if result.success | (result.iterations >= maxiter)
        p = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]

        if approach == :poisson
            result.whs = geo_weights(result.beta, prob.wh, prob.xmat, size(prob.geotargets))
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
