function geosolve2(prob::GeoweightProblem)
    print("Solving problem...")
    # function fvec2(beta::Vector{Float64})
    #     # beta = reshape(beta, size(geotargets))
    #     objvec(beta, prob.wh, prob.xmat, prob.geotargets)
    # end
    # function fvec2(beta::Vector{Float64})
    #     # beta = reshape(beta, size(geotargets))
    #     objvec(beta, wh, xmat, geotargets)
    # end
    # wh = prob.wh
    # xmat = prob.xmat
    # geotargets = prob.geotargets
    # ibeta = zeros(length(prob.geotargets))

    # lsres = getres(ibeta, prob.wh, prob.xmat, prob.geotargets)

    # include("functions_poisson.jl")
    # println(wh)
    # println(xmat)
    # println(geotargets)
    # println(ibeta)
    # f = beta -> objvec(beta, wh, xmat, geotargets)



    # r = f(ibeta)
    # r2 = OnceDifferentiable(f, ibeta, copy(r); inplace = false, autodiff = :forward)
    # lsres = LsqFit.levenberg_marquardt(r2, ibeta, show_trace = true)

    # return f(ibeta)
    # println(prob.wh)
    # println(size(prob.xmat))
    # println(size(prob.geotargets))
    # ibeta = zeros(length(prob.geotargets))
    # println(f(ibeta))
    # using Optim, LsqFit, Dates
    # r = OnceDifferentiable(f, ibeta; autodiff=true)
    # return r


    # f2(x) = x.^4 .- x.^2
    # f2([1., 2., 3.])

    # GOOD:
    # x0 = [1., 2., 3.]
    # r = f2(x0)
    # r2 = OnceDifferentiable(f2, x0, copy(r); inplace = false, autodiff = :forward)
    # lsres = LsqFit.levenberg_marquardt(r2, x0, show_trace = true)
    # lsres = LsqFit.lmfit(f2, [1., 2., 3.], Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=50)
    #LsqFit.lmfit(fvec!, ibeta, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=50)
    #lsres = LsqFit.levenberg_marquardt(r, ibeta)

    # # lsres = LsqFit.lmfit(f, ibeta, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=50)
    return lsres
end

function harness(x, y, z)
    f10 = x -> f1(x, y, z)
    LsqFit.lmfit(f10, x, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=50)
end

function harness_a(x, y, z)
    f10 = x -> f1(x, y, z)
    r = f10(x)
    r2 = OnceDifferentiable(f10, x, copy(r); inplace = false, autodiff = :forward)
    LsqFit.levenberg_marquardt(r2, x, show_trace = true, maxIter=50)
    # LsqFit.lmfit(f10, x, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=50)
end

function harness2(beta, wh, xmat, geotargets)
    f10 = beta -> objvec(beta, wh, xmat, geotargets)
    LsqFit.lmfit(f10, beta, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=50)
end

function harness2_a(beta, wh, xmat, geotargets)
    f10 = beta -> objvec(beta, wh, xmat, geotargets)
    r = f10(beta)
    r2 = OnceDifferentiable(f10, beta, copy(r); inplace = false, autodiff = :forward)
    LsqFit.levenberg_marquardt(r2, beta, show_trace = true, maxIter=50)
    # LsqFit.lmfit(f10, x, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=50)
end


# function harness3_a(beta, prob)
#     f10 = beta -> objvec(beta, prob.wh, prob.xmat, prob.geotargets)
#     r = f10(beta)
#     r2 = OnceDifferentiable(f10, beta, copy(r); inplace = false, autodiff = :forward)
#     res = LsqFit.levenberg_marquardt(r2, beta, show_trace = true, maxIter=50)
#     return res, f10(res.minimizer)
#     # LsqFit.lmfit(f10, x, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=50)
# end

function geosolve(prob, method; beta0=zeros(length(prob.geotargets)),
    maxiter=100, scaling=false, scaling_target_goal=10.0,
    kwargs...)
    # allowable methods:
    #   lm_lsqfit, lm_minpack
    println("Solving problem...")
    result = Result(method=method)

    tstart = time()
    if scaling
        prob.wh_scaled = prob.wh
        # targscale = targ_goal ./ maximum(abs.(tp.geotargets), dims=1)
        # targscale = targ_goal ./ sum(tp.geotargets, dims=1)
        scaling_target_factor = scaling_target_goal ./ sum(prob.xmat, dims=1)
        prob.geotargets_scaled = scaling_target_factor .* prob.geotargets
        prob.xmat_scaled = scaling_target_factor .* prob.xmat
    else
        prob.wh_scaled = prob.wh
        prob.geotargets_scaled = prob.geotargets
        prob.xmat_scaled = prob.xmat
    end

    result.problem = prob

    if method == :lm_lsoptim
        lsoptim(prob, beta0, result; maxiter=maxiter, kwargs...)
    elseif method == :lm_lsqfit
        lsqlm(prob, beta0, result; maxiter=maxiter, kwargs...)
    elseif method == :lm_minpack
        minpack(prob, beta0, result; maxiter=maxiter, kwargs...)
    elseif method == :lm_mads
        mads(prob, beta0, result; maxiter=maxiter, kwargs...)
    else
        error("Unknown method!")
        return;
    end
    tend = time()
    result.eseconds = tend - tstart

    if result.success
        p = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
        result.whs = geo_weights(result.beta, prob.wh, prob.xmat, size(prob.geotargets))
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
