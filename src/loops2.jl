# module fn2

function whs2(wh, mshares)
    # note that mshares = reshape(shares, length(wh), :)
    # same as whs = wh .* mshares
    hs = size(mshares)
    whs = similar(mshares)

    for i in 1:hs[1]
    for j in 1:hs[2]
        whs[i, j] = wh[i] * mshares[i, j]
    end
    end
    whs
end

function calctargets2(whs, xmat)
    # same as calctargets = whs' * xmat
    h, s = size(whs)
    k = size(xmat)[2]
    calctargets = zeros(s, k)

    for is in 1:s
        for ik in 1:k
            for ih in 1:h
                calctargets[is, ik] = calctargets[is, ik] + whs[ih, is] * xmat[ih, ik]
            end
        end
    end
    calctargets
end

function pdiffs2(calctargets, geotargets)
    # same as ...
    s, k = size(geotargets)
    pdiffs = zeros(s, k)

    for is in 1:s
        for ik in 1:k
        pdiffs[is, ik] = (calctargets[is, ik] - geotargets[is, ik]) / geotargets[is, ik] * 100.
        end
    end
    pdiffs
end

function sspdfn(pdiffs)
    sspd = 0.0
    s, k = size(pdiffs)
    for is in 1:s
      for ik in 1:k
        sspd = sspd + pdiffs[is, ik]^2
      end
    end
    sspd
end

function fopt2(shares, wh, xmat, geotargets)
    # shares: vec of (h * s)
    h = length(wh)
    s, k = size(geotargets)
    mshares = reshape(shares, h, s) # shares, as a matrix

    # whs = zeros(length(wh), size(geotargets)[1]) # preallocate
    # calctargets = zeros(size(geotargets)) # preallocate
    # pdiffs = zeros(size(geotargets)) # preallocate

    whs = whs2(wh, mshares)
    calctargets = calctargets2(whs, xmat)
    pdiffs = pdiffs2(calctargets, geotargets)

    return sspdfn(pdiffs)
end

# end