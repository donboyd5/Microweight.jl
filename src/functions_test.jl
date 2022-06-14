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


function f1(x, y, z)
    x .* y  .+ z
end

# f1([1., 2., 3.], 7., 9.)

function harness3_a(ibeta, prob)
    println("this is it yet again")
    f = beta -> objvec(beta, prob.wh, prob.xmat, prob.geotargets)
    f_init = f(ibeta)
    od = OnceDifferentiable(f, ibeta, copy(f_init); inplace = false, autodiff = :forward)
    res = LsqFit.levenberg_marquardt(od, ibeta, show_trace = true, maxIter=50)
    return res, f(res.minimizer)
    # LsqFit.lmfit(f10, x, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=50)
end


