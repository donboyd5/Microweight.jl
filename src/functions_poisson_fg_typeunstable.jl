
## objective functions ---------------------------

function f(beta)
  objfn(beta, wh, xmat, geotargets) # / obj_scale
end

function f!(out, beta)
  out = objfn(beta, wh, xmat, geotargets) # / obj_scale
end

function g!(G, x)
  G .=  f'(x)
end

function f2(beta, pdummy)
  # pdummy added for GalacticOptim
  objfn(beta, wh, xmat, geotargets)
end

## gradient functions ------------------------------
function fvec(beta)
  # beta = reshape(beta, size(geotargets))
  objvec(beta, wh, xmat, geotargets)
end

function fvec!(out, beta)
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

