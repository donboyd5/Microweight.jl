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

#                            objval, targ_rmse, targpdiffs, ratio_rmse, ratiodiffs, targstop
function cb_rwminerr2(ratio, objval)#, targ_rmse, targpdiffs, ratio_rmse, ratiodiffs, targstop)
    println("objval: $objval")
    objval < 0.00065
end

function cb_rwminerr(ratio, objval, targ_rmse, targpdiffs, ratio_rmse, ratiodiffs, targstop)
    # list extra variables on the return so that they are available to the callback function
    # all returned variables must be arguments of the callback function
    # return objval, targpdiffs, ratiodiffs # values to be used in callback function must be returned here
    # values other than ratio and objval that are to be used in the callback function must be returned from the objective function

   # println("about to stop")
    return true
    println("still going")

    # declare as global any variables that must persist from one call to the next, and may be changed in the callback
    # initial values are set in rwsolve() in api.jl 
    global fcalls  # init val 0
    global bestobjval  # init val Inf
    global iter_calc  # init val 0

    # println("targstop is $targstop")

    fcalls += 1
    new_iter = false
  
    if objval < bestobjval || (fcalls<=5 && objval > bestobjval)
        bestobjval = objval
        iter_calc += 1
        new_iter = true
        use_iter = mod(iter_calc, interval) == 0 || iter_calc in (0:2)
    end

    if !new_iter return false end # don't bother to check stopping criteria

    # get statistics for targets
    targ_max = maximum(abs.(targpdiffs))
    targ_ptile = Statistics.quantile!(vec(abs.(targpdiffs)), plevel)
    #  targ_rmse=targ_rmse,  wtsum_rmse=wtsum_rmse, tot_rmse=tot_rmse,

    # get statistics for ratios
    ratio_max = maximum(abs.(ratiodiffs))
    ratio_ptile = Statistics.quantile!(vec(abs.(ratiodiffs)), plevel)

    totseconds = time() - tstart

    if use_iter 
        rwshow_iter(iter_calc=iter_calc, fcalls=fcalls, totseconds=totseconds, objval=objval,
                targ_rmse=targ_rmse, targ_max=targ_max, targ_ptile=targ_ptile,
                ratio_rmse=ratio_rmse, ratio_max=ratio_max, ratio_ptile=ratio_ptile)
    end
  
    # halt = targ_max < targstop && wtsum_max < whstop

    println("targ_max: $targ_max and targstop: $targstop")
    halt = targ_max <= targstop
    println("halt: $halt")
    # halt = false
    return true
  end


function rwshow_iter(; iter_calc, fcalls, totseconds, objval, 
    targ_rmse, targ_max, targ_ptile,
    ratio_rmse, ratio_max, ratio_ptile)
    # declare as global any variables that must persist from one call to the next, and may be changed
    global nshown  # init val 0
    # print headings if needed
    nshown += 1
    if nshown == 1 || mod(nshown, 20) == 0
        println()
        hdr1 = "iter_calc   fcalls  totseconds       objval    targ_rmse     targ_max"
        hdr2 = "      "
        hdr3 = "targ_" * string(floor(Int, plevel * 100.))
        hdr4 = "  ratiod1_rmsd  ratiod1_max   "
        hdr5 = "ratiod1_" * string(floor(Int, plevel * 100.))
        hdr = hdr1 * hdr2 * hdr3 * hdr4 * hdr5
        println(hdr)
    end

    # display results
    @printf(" %8i %8i %11.3g %12.6g %12.4g %12.4g %12.4g  %12.4g %12.4g %12.4g \n",
    iter_calc, fcalls, totseconds, objval, targ_rmse, targ_max, targ_ptile, ratio_rmse, ratio_max, ratio_ptile)
    return
end


function cb_spg(R::SPGBoxResult, wh, xmat, rwtargets)
    # The spg callback function has as input the SPGBoxResult structure.
    # See this: https://m3g.github.io/SPGBox.jl/stable/usage/#Result-data-structure-and-possible-outcomes
    # struct: x vector(Float64), f, gnorm Float64, nit, nfeval, ierr Int64, return_from_callback::Bool
    # The callback must return true or false. If true, spgbox will return immediately with the current point.

    # if R.nit <= 1
    #     return false 
    # else
    #     return true
    # end
    
    # println("spg callback 11")
    use_iter = true

    targ_rmse, targpdiffs, ratio_rmse, ratiodiffs = getvals(R.x, wh, xmat, rwtargets)

    # get statistics for targets
    targ_max = maximum(abs.(targpdiffs))
    targ_ptile = Statistics.quantile!(vec(abs.(targpdiffs)), plevel)
    #  targ_rmse=targ_rmse,  wtsum_rmse=wtsum_rmse, tot_rmse=tot_rmse,

    # get statistics for ratios
    ratio_max = maximum(abs.(ratiodiffs))
    ratio_ptile = Statistics.quantile!(vec(abs.(ratiodiffs)), plevel)  

    totseconds = time() - tstart

    if use_iter 
        rwshow_iter(iter_calc=R.nit, fcalls=R.nfeval, totseconds=totseconds, objval=R.f,
                targ_rmse=targ_rmse, targ_max=targ_max, targ_ptile=targ_ptile,
                ratio_rmse=ratio_rmse, ratio_max=ratio_max, ratio_ptile=ratio_ptile)
    end

    halt = false
    return halt
end

cb_spg(R) = cb_spg(R, wh, xmat, rwtargets)  # ::SPGBoxResult

function getvals(ratio, wh, xmat, rwtargets)
  # part 1 get measure of difference from targets
  rwtargets_calc = xmat' * (ratio .* wh)
  targpdiffs = (rwtargets_calc .- rwtargets) ./ rwtargets # ./ 1e6 # allocates a tiny bit
  targ_sse = sum(targpdiffs.^2.)
  targ_rmse = targ_sse / length(targpdiffs)

  # part 2 - measure of change in ratio
  ratiodiffs = ratio .- 1.0
  ratio_sse = sum(ratiodiffs.^2.)
  ratio_rmse = ratio_sse / length(ratiodiffs)

  return targ_rmse, targpdiffs, ratio_rmse, ratiodiffs
end



## functions used for testing purposes below here #################################################################################

function cb_test(ratio, objval, targ_rmse, targpdiffs, ratio_rmse, ratiodiffs)
    println("callback 2")
    global fcalls  # init val 0
    global bestobjval  # init val Inf
    global iter_calc  # init val 0

    fcalls += 1
    new_iter = false

    if objval < bestobjval || (fcalls<=5 && objval > bestobjval)
        bestobjval = objval
        iter_calc += 1
        new_iter = true
        use_iter = mod(iter_calc, interval) == 0 || iter_calc in (0:2)
    end     

    if !new_iter return false end # don't bother to check stopping criteria

    # get statistics for targets
    targ_max = maximum(abs.(targpdiffs))
    targ_ptile = Statistics.quantile!(vec(abs.(targpdiffs)), plevel)
    #  targ_rmse=targ_rmse,  wtsum_rmse=wtsum_rmse, tot_rmse=tot_rmse,

    # get statistics for ratios
    ratio_max = maximum(abs.(ratiodiffs))
    ratio_ptile = Statistics.quantile!(vec(abs.(ratiodiffs)), plevel)

    totseconds = time() - tstart    

    if use_iter 
        rwshow_iter2(iter_calc=iter_calc, fcalls=fcalls, totseconds=totseconds, objval=objval,
                targ_rmse=targ_rmse, targ_max=targ_max, targ_ptile=targ_ptile,
                ratio_rmse=ratio_rmse, ratio_max=ratio_max, ratio_ptile=ratio_ptile)
    end
    
    return false
end

function rwshow_iter2(; iter_calc, fcalls, totseconds, objval, 
    targ_rmse, targ_max, targ_ptile,
    ratio_rmse, ratio_max, ratio_ptile)
    println("rshow 3")

    global nshown  # init val 0
    # print headings if needed
    nshown += 1
    if nshown == 1 || mod(nshown, 20) == 0
        println()
        hdr1 = "iter_calc   fcalls  totseconds       objval    targ_rmse     targ_max"
        hdr2 = "      "
        hdr3 = "targ_" * string(floor(Int, plevel * 100.))
        hdr4 = "    ratio_rmse   ratio_max    "
        hdr5 = "ratio_" * string(floor(Int, plevel * 100.))
        hdr = hdr1 * hdr2 * hdr3 * hdr4 * hdr5
        println(hdr)
    end

    # display results
    @printf(" %8i %8i %11.3g %12.6g %12.4g %12.4g %12.4g %12.4g %12.4g %12.4g \n",
    iter_calc, fcalls, totseconds, objval, targ_rmse, targ_max, targ_ptile, ratio_rmse, ratio_max, ratio_ptile)    

    return
end



