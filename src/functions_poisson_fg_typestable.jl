
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
    objvec2(beta, wh, xmat, geotargets)
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
