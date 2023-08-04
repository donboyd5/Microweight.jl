# In the package Optimizers.jl:
# https://docs.sciml.ai/Optimization/stable/API/solve/#CommonSolve.solve-Tuple{OptimizationProblem,%20Any}
# Callback Functions
# The callback function callback is a function which is called after every optimizer step. Its signature is:

#     callback = (params, loss_val, other_args) -> false
    
# where params and loss_val are the current parameters and loss/objective value in the optimization loop and other_args are 
# the extra return arguments of the optimization f. This allows for saving values from the optimization and using them for plotting
# and display without recalculating. The callback should return a Boolean value, and the default should be false, such that
# the optimization gets stopped if it returns true.

# what I did for geosolve:
# fp = (shares, p) -> objfn_direct(shares, prob.wh_scaled, prob.xmat_scaled, prob.geotargets_scaled,
#       p_mshares, p_whs, p_calctargets, p_pdiffs, p_whpdiffs, whweight, pow, targstop, whstop)
# cb_direct(shares, objval, p_pdiffs, p_whpdiffs, targstop, whstop)

# what I have for reweight:
# fp = (ratio, p) -> objfn_reweight(ratio, wh, xmat, rwtargets, rweight=rweight)
# cb_rwminerr(shares, objval, p_pdiffs, p_whpdiffs, targstop, whstop)



function cb_test(ratio, objval, targpdiffs, ratiodiffs)
    # list extra variables on the return so that they are available to the callback function
    # all returned variables must be arguments of the callback function
    # return objval, targpdiffs, ratiodiffs # values to be used in callback function must be returned here
    # values other than ratio and objval that are to be used in the callback function must be returned from the objective function
    # declare as global any variables that must persist from one call to the next, and may be changed
    global fcalls  # init val 0

    fcalls += 1

    if fcalls < 5
        println("fcalls: $fcalls")
        println("objval: $objval")
        println("ratio quantile: $(quantile(ratio))")
        println("targpdiffs quantile: $(quantile(targpdiffs))")        
        println("ratiodiffs quantile: $(quantile(ratiodiffs))")        
    end
  
    halt = false
    return halt
  end


function cb_rwminerr(shares, objval, p_pdiffs, p_whpdiffs, targstop, whstop)
    # declare as global any variables that must persist from one call to the next, and may be changed
    global fcalls  # init val 0
    global bestobjval  # init val Inf
    global iter_calc  # init val 0
  
    fcalls += 1
    new_iter = false
  
    if objval < bestobjval || (fcalls<=5 && objval > bestobjval)
        bestobjval = objval
        iter_calc += 1
        new_iter = true
        use_iter = mod(iter_calc, interval) == 0 || iter_calc in (0, 1)
    end
  
    if !new_iter return false end # don't bother to check stopping criteria
  
    # get statistics for targets
    targ_max = maximum(abs.(p_pdiffs))
    targ_ptile = Statistics.quantile!(vec(abs.(p_pdiffs)), plevel)
  
    # get statistics for weights
    wtsum_max = maximum(abs.(p_whpdiffs))
    wtsum_ptile = Statistics.quantile!(vec(abs.(p_whpdiffs)), plevel)
  
    targ_sse = sum(p_pdiffs.^2)
    wtsum_sse = sum(p_whpdiffs.^2)
  
    targ_rmse = sqrt(targ_sse / length(p_pdiffs))
    wtsum_rmse = sqrt(wtsum_sse / length(p_whpdiffs))
    tot_rmse = sqrt((targ_sse + wtsum_sse) / (length(p_pdiffs) + length(p_whpdiffs)))
  
    totseconds = time() - tstart
  
    if use_iter rwshow_iter(iter_calc=iter_calc, fcalls=fcalls, totseconds=totseconds,
        objval=objval,
        targ_rmse=targ_rmse,  wtsum_rmse=wtsum_rmse, tot_rmse=tot_rmse,
        targ_max=targ_max, wtsum_max=wtsum_max,
        targ_ptile=targ_ptile, wtsum_ptile=wtsum_ptile)
    end
  
    halt = targ_max < targstop && wtsum_max < whstop
    return halt
  end
  

function rwshow_iter(; iter_calc, fcalls, totseconds, objval, targ_rmse,  wtsum_rmse, tot_rmse, targ_max, wtsum_max, targ_ptile, wtsum_ptile)
    # declare as global any variables that must persist from one call to the next, and may be changed
    global nshown  # init val 0
    # print headings if needed
    nshown += 1
    if nshown == 1 || mod(nshown, 20) == 0
        println()
        hdr1 = "iter_calc   fcalls  totseconds       objval    targ_rmse   wtsum_rmse     tot_rmse     targ_max    wtsum_max"
        hdr2 = "      "
        hdr3 = "targ_" * string(floor(Int, plevel * 100.))
        hdr4 = "     "
        hdr5 = "wtsum_" * string(floor(Int, plevel * 100.))
        hdr = hdr1 * hdr2 * hdr3 * hdr4 * hdr5
        println(hdr)
    end

    # display results
    @printf(" %8i %8i %11.5g %12.7g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g \n",
    iter_calc, fcalls, totseconds, objval, targ_rmse,  wtsum_rmse, tot_rmse, targ_max, wtsum_max, targ_ptile, wtsum_ptile)
    return
end

