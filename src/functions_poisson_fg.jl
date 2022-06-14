# using LsqFit

## objective functions ---------------------------

function f(beta::Vector{Float64})
    objfn(beta, wh, xmat, geotargets) # / obj_scale
end

function f!(out, beta::Vector{Float64})
    out = objfn(beta, wh, xmat, geotargets)
end

function g!(G, beta::Vector{Float64})
    G .=  f'(beta)
end

function f2(beta::Vector{Float64}, pdummy)
    # pdummy added for GalacticOptim
    objfn(beta, wh, xmat, geotargets)
end

## gradient functions ------------------------------

function fvec(beta::Vector{Float64})
    # beta = reshape(beta, size(geotargets))
    objvec(beta, wh, xmat, geotargets)
end

function fvec!(out, beta::Vector{Float64})
    # for LeastSquaresOptim inplace
    out .= objvec(beta, wh, xmat, geotargets)
end

## jacobian function ----

function gvec(beta)
    ForwardDiff.jacobian(x -> fvec(x), beta)
 end

function gvec!(out, beta)
  out .= ForwardDiff.jacobian(x -> fvec(x), beta)
end

function getres(ibeta, wh, xmat, geotargets)
    wh = copy(wh)
    xmat = copy(xmat)
    geotargets = copy(geotargets)
    function fvecz(beta::Vector{Float64})
        # beta = reshape(beta, size(geotargets))
        println("in fvecz, wh:")
        println(wh)
        objvec(beta, wh, xmat, geotargets)
    end
    # r = fvecz(ibeta)
    # println("out of fvecz, r (fvecz at ibeta):")
    # println(r)
    # println("about to do r2")
    # r2 = OnceDifferentiable(fvecz, ibeta, copy(r); inplace = false, autodiff = :forward)
    # println("done r2")
    # return r2
    lsres = LsqFit.lmfit(fvecz, copy(ibeta), Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=10)
    return lsres
    lsres = LsqFit.levenberg_marquardt(r2, ibeta, show_trace = true)
    return lsres
end




