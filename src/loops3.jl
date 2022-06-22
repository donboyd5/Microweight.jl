module loop

global fcalls

function whs(wh, mshares, p)
    # note that mshares = reshape(shares, length(wh), :)
    # same as whs = wh .* mshares
    # whs h, s is preallocated

    for i in 1:p.h
        for j in 1:p.s
            p.whs[i, j] = wh[i] * mshares[i, j]
        end
    end
    p.whs
end

function calctargets(whs, xmat, p)
    # same as calctargets = whs' * xmat
    # calctargets s, k is preallocated
    p.calctargets = zeros(p.s, p.k)
    for is in 1:p.s
        for ik in 1:p.k
            for ih in 1:p.h
                p.calctargets[is, ik] = p.calctargets[is, ik] + whs[ih, is] * xmat[ih, ik]
            end
        end
    end
    p.calctargets
end

function pdiffs(calctargets, geotargets, p)
    # pdiffs s, k is preallocated
    p.pdiffs = zeros(p.s, p.k)
    for is in 1:p.s
        for ik in 1:p.k
            p.pdiffs[is, ik] = (calctargets[is, ik] - geotargets[is, ik]) / geotargets[is, ik] * 100.
        end
    end
    p.pdiffs
end

function sspdfn(pdiffs, p)
    p.sspd = 0.0
    for is in 1:p.s
      for ik in 1:p.k
        p.sspd = p.sspd + pdiffs[is, ik]^2
      end
    end
    p.sspd
end

function fopt(shares, wh, xmat, geotargets, p)
    # shares: vec of (h * s)
    # wh, xmat, and geotargets are constants
    # mshares, whs, calctargets, and pdiffs are mutable but preallocated
    # global fcalls += 1
    p.mshares = reshape(shares, p.h, p.s)
    p.whs = whs(wh, p.mshares, p)
    p.calctargets = calctargets(p.whs, xmat, p)
    p.pdiffs = pdiffs(p.calctargets, geotargets, p)
    p.sspd = sspdfn(p.pdiffs, p)
    # println("fcall: ", fcalls, " sspd: ", p.sspd)
    return p.sspd
end

end