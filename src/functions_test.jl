# using LsqFit

function lsq(ibeta::Vector{Float64}, wh::Matrix{Float64}, xmat::Matrix{Float64}, geotargets::Matrix{Float64})
    println("hello")
    # fx(beta) = objvec(beta, wh, xmat, geotargets)
    # fx = beta -> objvec(beta, wh, xmat, geotargets)

    function fvec!(out, beta)
        # for LeastSquaresOptim inplace
        out .= objvec(beta, wh, xmat, geotargets)
    end
    # fx! = beta -> fvec!(out, beta)
    # fx! = beta -> .= objvec(out, beta, wh, xmat, geotargets)
    # z = fx(ibeta)
    # print(z)
    # fvec(ibeta)
    LsqFit.lmfit(fvec!, ibeta, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=50)
    # 136.778423 seconds (8.19 M allocations: 419.210 GiB, 9.40% gc time, 1.11% compilation time)
  end