
function geosolve(prob;
    approach=nothing,
    method=nothing,
    beta0=nothing,
    shares0=nothing,
    maxiter=100,
    objscale=1.0, scaling=false, scaling_target_goal=1000.0,
    interval=1,
    whweight=nothing,
    kwargs...)
    # allowable methods:
    #   lm_lsqfit, lm_minpack
    println("Solving problem...")

    global tstart = time()
    global fcalls = 0  # global within this module
    global h = prob.h
    global s = prob.s
    global k = prob.k
    global objdiv = 100.
    global pow = 4
    global s_scale = 1e0
    global plevel = .99
    global whweight2
    global bestobjval = 1e99
    global nshown = 0
    global iter_calc = 0

    # define defaults
    if isnothing(approach) approach=:poisson end

    if approach==:poisson
        if isnothing(method) method=:lm_lsqfit end
        if isnothing(beta0) beta0 = zeros(length(prob.geotargets)) end
    elseif approach==:direct
        if isnothing(method) method=:direct_test2 end
        if isnothing(shares0) shares0=fill(1. / prob.s, prob.h * prob.s) end
    else
        return "ERROR: approach must be :poisson or :direct"
    end

    # initialize result
    prob = scale_prob(prob, scaling=scaling, scaling_target_goal=scaling_target_goal)
    result = Result(approach=approach, method=method, problem=prob, beta0=beta0, shares0=shares0)

    if approach == :poisson
        if method == :cg_optim
            poisson_cgoptim(prob, beta0, result; maxiter=maxiter, objscale=objscale, interval, kwargs...)
        elseif method == :cg_optim2
            poisson_cgoptim2(prob, beta0, result; maxiter=maxiter, objscale=objscale, interval, kwargs...)
        elseif method == :lm_lsoptim
            poisson_lsoptim(prob, beta0, result; maxiter=maxiter, objscale=objscale, interval, kwargs...)
        elseif method == :lm_lsqfit
            poisson_lsqlm(prob, beta0, result; maxiter=maxiter, objscale=objscale, interval, kwargs...)
        elseif method == :lm_minpack
            poisson_minpack(prob, beta0, result; maxiter=maxiter, objscale=objscale, interval, kwargs...)
        # elseif method == :lm_mads
        #     poisson_mads(prob, beta0, result; maxiter=maxiter, objscale=objscale, kwargs...)
        elseif method == :newttr_nlsolve
            poisson_newttrust(prob, beta0, result; maxiter=maxiter, objscale=objscale, interval, kwargs...)
        elseif method == :krylov
            poisson_krylov(prob, beta0, result; maxiter=maxiter, objscale=objscale, interval, kwargs...)
        else
            error("Unknown poisson method!")
            return;
        end
    elseif approach == :direct
        okmethod = (:direct_cg, :direct_krylov)
        println("goodmethod = ", method in okmethod)

        if method==:direct_cg
            direct_cg(prob, result; whweight=nothing, maxiter=maxiter, interval)
        elseif method == :direct_krylov
            direct_krylov(prob, shares0, result; whweight=nothing, maxiter=maxiter, interval)
        elseif method == :direct_test
            direct_test(prob, shares0, result; whweight=nothing, maxiter=maxiter, interval)
        elseif method == :direct_test2
            direct_test_scaled(prob, shares0, result, maxiter=maxiter, interval=interval, whweight=whweight)
        # elseif method == :direct_krylov_bounds
        #     direct_krylov_bounds(prob, shares0, result; whweight=nothing, maxiter=maxiter, interval)
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
