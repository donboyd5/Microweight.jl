
function geo_weights(beta, wh, xmat, targshape)
    # beta: coefficients, s x k
    # wh: weight for each household, h values
    # xmat: hh characteristics, h x k
    # betax = beta.dot(xmat.T)

    # bdims = size(geotargets)
    beta = reshape(beta, targshape)

    betaxp = beta * xmat'  # (s x k) * (k x h) = s x h

    # adjust betax to make exponentiation more stable numerically
    # subtract column-specific constant (the max) from each column of betax
    # const = betax.max(axis=0)
    betaxpmax = maximum(betaxp, dims=1) # max of each col of betaxp: 1 x h
    # betax = jnp.subtract(betax, const)
    betaxpadj = betaxp .- betaxpmax # (s x h) - (1 x h) = (s x h)
    ebetaxpadj = exp.(betaxpadj) # s x h
    # logdiffs = betax - jnp.log(ebetax.sum(axis=0))
    colsums = sum(ebetaxpadj, dims=1)  # 1 x h
    logdiffs = betaxpadj .- log.(colsums) # (s x h) - (1 x h) = (s x h)
    shares = exp.(logdiffs)' # (h x s) after transpose
    whs = wh .* shares # (h x 1) x (h x s) elementwise
    whs
end

function geo_targets(whs, xmat)
    whs' * xmat
end

function targ_pdiffs(calctargets, geotargets)
    diffs = calctargets - geotargets
    pdiffs = diffs ./ geotargets * 100.
    pdiffs
end

function sspd(calctargets, geotargets)
    # worry about what to do when a geotarget is zero
    pdiffs = targ_pdiffs(calctargets, geotargets)
    sqpdiffs = pdiffs.^2
    sspd = sum(sqpdiffs)
    sspd
end

function objfn(beta, wh, xmat, geotargets)
    # beta = reshape(beta, )
    targshape = size(geotargets)
    whs = geo_weights(beta, wh, xmat, targshape)
    calctargets = geo_targets(whs, xmat)
    obj = sspd(calctargets, geotargets)
    obj
end

function objvec(beta, wh, xmat, geotargets)
    targshape = size(geotargets)
    beta = reshape(beta, targshape)
    whs = geo_weights(beta, wh, xmat, targshape)
    calctargets = geo_targets(whs, xmat)
    objvec = targ_pdiffs(calctargets, geotargets)
    vec(objvec)
end

function objvec!(out, beta, wh, xmat, geotargets)
    targshape = size(geotargets)
    beta = reshape(beta, targshape)
    whs = geo_weights(beta, wh, xmat, targshape)
    calctargets = geo_targets(whs, xmat)
    objvec = targ_pdiffs(calctargets, geotargets)
    out .= vec(objvec)
end

function objvec2(beta, wh, xmat, geotargets, fcalls, interval=1)
    # global fcalls
    targshape = size(geotargets)
    beta = reshape(beta, targshape)
    whs = geo_weights(beta, wh, xmat, targshape)
    calctargets = geo_targets(whs, xmat)
    objvec = targ_pdiffs(calctargets, geotargets)

    display_progress = true
    if display_progress
        display1(fcalls, interval, geotargets, calctargets, wh, whs)
    end

    return vec(objvec)
end