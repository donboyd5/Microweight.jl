module loop

function whs3(wh, mshares, h, s, whs)
    # note that mshares = reshape(shares, length(wh), :)
    # same as whs = wh .* mshares
    # whs h, s is preallocated
    for i in 1:h
        for j in 1:s
            whs[i, j] = wh[i] * mshares[i, j]
        end
    end
    whs
end

function calctargets3(whs, xmat, h, k, s, calctargets)
    # same as calctargets = whs' * xmat
    # calctargets s, k is preallocated
    for is in 1:s
        for ik in 1:k
            for ih in 1:h
                calctargets[is, ik] = calctargets[is, ik] + whs[ih, is] * xmat[ih, ik]
            end
        end
    end
    calctargets
end

function pdiffs3(calctargets, geotargets, k, s, pdiffs)
    # pdiffs s, k is preallocated
    for is in 1:s
        for ik in 1:k
        pdiffs[is, ik] = (calctargets[is, ik] - geotargets[is, ik]) / geotargets[is, ik] * 100.
        end
    end
    pdiffs
end

function sspdfn3(pdiffs, k, s)
    sspd = 0.0
    for is in 1:s
      for ik in 1:k
        sspd = sspd + pdiffs[is, ik]^2
      end
    end
    sspd
end

function fopt3(shares, wh, xmat, geotargets, h, k, s, mshares, whs, calctargets, pdiffs)
    # shares: vec of (h * s)
    # wh, xmat, and geotargets are constants
    # mshares, whs, calctargets, and pdiffs are mutable but preallocated

    mshares = reshape(shares, h, s) # shares, as a matrix
    whs = whs3(wh, mshares, h, s, whs)
    calctargets = calctargets3(whs, xmat, h, k, s, calctargets)
    pdiffs = pdiffs3(calctargets, geotargets, k, s, pdiffs)
    sspd = sspdfn3(pdiffs, k, s)
    return sspd
end

end