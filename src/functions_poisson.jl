
function geo_weights(beta::Vector{Float64}, wh::Matrix{Float64}, xmat::Matrix{Float64}, targshape::Tuple{Int64, Int64})
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

function geo_targets(whs::Matrix{Float64}, xmat::Matrix{Float64})
    whs' * xmat
end

function targ_pdiffs(calctargets::Matrix{Float64}, geotargets::Matrix{Float64})
    diffs = calctargets - geotargets
    pdiffs = diffs ./ geotargets * 100.
    pdiffs
end

function objfn(beta::Vector{Float64}, wh::Matrix{Float64}, xmat::Matrix{Float64}, geotargets::Matrix{Float64})
    # beta = reshape(beta, )
    targshape = size(geotargets)
    whs = geo_weights(beta, wh, xmat, targshape)
    calctargets = geo_targets(whs, xmat)
    obj = sspd(calctargets, geotargets)
    obj
end

function objvec(beta::Vector{Float64}, wh::Matrix{Float64}, xmat::Matrix{Float64}, geotargets::Matrix{Float64})
    targshape = size(geotargets)
    # beta = reshape(beta, targshape)
    whs = geo_weights(beta, wh, xmat, targshape)
    calctargets = geo_targets(whs, xmat)
    objvec = targ_pdiffs(calctargets, geotargets)
    vec(objvec)
end

# f2(x) = x.^4 .- x.^2
# f3 = x -> f2(x)

# f = beta -> objvec(beta, wh, xmat, geotargets)

