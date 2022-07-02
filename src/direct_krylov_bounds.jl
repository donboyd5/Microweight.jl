function direct_krylov_bounds(prob, shares0, result; maxiter=100, objscale=1.0, interval=1, whweight, kwargs...)

    kwkeys_allowed = (:show_trace, :x_tol, :g_tol)
    kwargs_keep = clean_kwargs(kwargs, kwkeys_allowed)

    # %% setup preallocations
    p = 1.0
    # shares0 = fill(1. / prob.s, prob.h * prob.s)
    p_mshares = Array{Float64,2}(undef, prob.h, prob.s)
    p_whs = Array{Float64,2}(undef, prob.h, prob.s)
    p_calctargets = Array{Float64,2}(undef, prob.s, prob.k)
    p_pdiffs = Array{Float64,2}(undef, prob.s, prob.k)
    p_whpdiffs = Array{Float64,1}(undef, prob.h)

    if whweight===nothing
        whweight = length(shares0) / length(p_calctargets)
    end
    println("Household weights component weight: ", whweight)

    f = shares -> objfn_direct(shares, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled,
          p_mshares, p_whs, p_calctargets, p_pdiffs, p_whpdiffs, interval, whweight)

    # optimize(_objective, l, u, initial_x, Fminbox(_optimizer))
# result = optimize(min_function, lx, ux, init_guess, Fminbox(LBFGS()))
    # opt = Optim.optimize(f, zeros(length(shares0)), ones(length(shares0)), shares0, Fminbox(LBFGS()))
    opt = Optim.optimize(f, zeros(length(shares0)), ones(length(shares0)), shares0, Fminbox(Optim.KrylovTrustRegion()))
# Optim.KrylovTrustRegion()

    result.solver_result = opt
    result.success = true
    result.iterations = -9
    result.shares = Optim.minimizer(opt)
    return result
end