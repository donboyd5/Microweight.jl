

function whs!(whs, wh, mshares)
    # whs: preallocate and pass
    # note that mshares = reshape(shares, length(wh), :)
    # same as whs = wh .* mshares
    hs = size(mshares)
    for i in 1:hs[1]
    for j in 1:hs[2]
        whs[i, j] = wh[i] * mshares[i, j]
    end
    end
    whs
end

function calctargets!(calctargets, whs, xmat)
    # preallocate calctargets = zeros(size(geotargets))
    # same as calctargets = whs' * xmat
    h, s = size(whs)
    k = size(xmat)[2]
    for is in 1:s
        for ik in 1:k
            for ih in 1:h
                calctargets[is, ik] = calctargets[is, ik] + whs[ih, is] * xmat[ih, ik]
            end
        end
    end
    calctargets
end

function pdiffs!(pdiffs, calctargets, geotargets)
    # preallocate pdiffs = similar(geotargets)
    # same as ...
    s, k = size(geotargets)
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

function fopt!(shares, wh, xmat, geotargets, whs, calctargets, pdiffs)
    # shares: vec of (h * s)
    h = length(wh)
    s, k = size(geotargets)
    mshares = reshape(shares, h, s) # shares, as a matrix

    # whs = similar(mshares) # preallocate
    whs = whs!(whs, wh, mshares)

    # calctargets = zeros(size(geotargets)) # preallocate
    calctargets = calctargets!(calctargets, whs, xmat)

    # pdiffs = similar(geotargets) # preallocate
    pdiffs = pdiffs!(pdiffs, calctargets, geotargets)

    return sspdfn(pdiffs)
end

