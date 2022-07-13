
function clean_kwargs(kwargs_passed, kwkeys_allowed)
    # returns a Dict that can be used as kwargs...
    # kwargs_passed::pairs(::NamedTuple), kwargs_allowed::pairs(::NamedTuple)
    filter(p -> first(p) in kwkeys_allowed, kwargs_passed)
end


# function display_status(interval, geotargets, p_calctargets, wh, p_whs, objval=nothing)
#     global fcalls  # init val 0
#     global nshown  # init val 0
#     global bestobjval  # init val Inf
#     global iter_calc  # init val 0

#     # global tstart
#     Zygote.ignore() do
#         fcalls += 1
#         new_iter = false

#         if objval < bestobjval || iter_calc in (0, 1)
#             bestobjval = objval
#             new_iter = true
#             iter_calc += 1
#         end

#         show_iter = mod(iter_calc, interval) == 0 || iter_calc in (0, 1)
#         show_iter = true

#         if new_iter && show_iter
#         # if true
#             nshown += 1

#             if nshown ==1 || mod(nshown, 20) == 0
#                 println()
#                 hdr1 = "iter_calc   fcalls  totseconds       objval    targ_rmse   wtsum_rmse     tot_rmse     targ_max    wtsum_max"
#                 hdr2 = "      "
#                 hdr3 = "targ_" * string(floor(Int, plevel * 100.))
#                 hdr4 = "     "
#                 hdr5 = "wtsum_" * string(floor(Int, plevel * 100.))
#                 hdr = hdr1 * hdr2 * hdr3 * hdr4 * hdr5
#                 println(hdr)
#             end

#             # get statistics for targets
#             p_pdiffs = (p_calctargets .- geotargets) ./ geotargets * 100.
#             targ_max = maximum(abs.(p_pdiffs))
#             targ_ptile = Statistics.quantile!(vec(abs.(p_pdiffs)), plevel)

#             # get statistics for weights
#             p_whpdiffs = (sum(p_whs, dims=2) .- wh) ./ wh * 100.
#             wtsum_max = maximum(abs.(p_whpdiffs))
#             wtsum_ptile = Statistics.quantile!(vec(abs.(p_whpdiffs)), plevel)

#             targ_sse = sum(p_pdiffs.^2)
#             wtsum_sse = sum(p_whpdiffs.^2)

#             targ_rmse = sqrt(targ_sse / length(p_pdiffs))
#             wtsum_rmse = sqrt(wtsum_sse / length(p_whpdiffs))
#             tot_rmse = sqrt((targ_sse + wtsum_sse) / (length(p_pdiffs) + length(p_whpdiffs)))

#             totseconds = time() - tstart

#             @printf(" %8i %8i %11.5g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g \n",
#               iter_calc, fcalls, totseconds, objval, targ_rmse,  wtsum_rmse, tot_rmse, targ_max, wtsum_max, targ_ptile, wtsum_ptile)
#         end
#     end
# end


function cb_direct(shares, objval, p_pdiffs, p_whpdiffs, targstop, whstop)
    # declare as global any variables that must persist from one call to the next, and may be changed
    global fcalls  # init val 0
    global nshown  # init val 0
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

    if use_iter show_iter(iter_calc=iter_calc, fcalls=fcalls, totseconds=totseconds,
        objval=objval,
        targ_rmse=targ_rmse,  wtsum_rmse=wtsum_rmse, tot_rmse=tot_rmse,
        targ_max=targ_max, wtsum_max=wtsum_max,
        targ_ptile=targ_ptile, wtsum_ptile=wtsum_ptile)
    end

    halt = targ_max < targstop && wtsum_max < whstop
    return halt
end

function cb_poisson(beta, objval, p_pdiffs, p_whs, wh, targstop, whstop)
    # objval, pdiffs, whs, wh, targstop, whstop
    # println("in cb_poisson ", objval)
    # return false
    # declare as global any variables that must persist from one call to the next, and may be changed
    global fcalls  # init val 0
    global nshown  # init val 0
    global bestobjval  # init val Inf
    global iter_calc  # init val 0

    fcalls += 1
    new_iter = false

    if objval < bestobjval || (fcalls<=5 && objval > bestobjval) || true
        bestobjval = objval
        new_iter = true
        iter_calc += 1
    end

    # show_iter = mod(iter_calc, interval) == 0 || iter_calc in (0, 1)
    show_iter = true

    if new_iter && show_iter
    # if true
        nshown += 1

        if nshown ==1 || mod(nshown, 20) == 0
            println()
            hdr1 = "iter_calc   fcalls  totseconds       objval    targ_rmse   wtsum_rmse     tot_rmse     targ_max    wtsum_max"
            hdr2 = "      "
            hdr3 = "targ_" * string(floor(Int, plevel * 100.))
            hdr4 = "     "
            hdr5 = "wtsum_" * string(floor(Int, plevel * 100.))
            hdr = hdr1 * hdr2 * hdr3 * hdr4 * hdr5
            println(hdr)
        end

        # get statistics for targets
        targ_max = maximum(abs.(p_pdiffs))
        targ_ptile = Statistics.quantile!(vec(abs.(p_pdiffs)), plevel)

        # get statistics for weights
        p_whpdiffs = (sum(p_whs, dims=2) .- wh) ./ wh * 100.
        wtsum_max = maximum(abs.(p_whpdiffs))
        wtsum_ptile = Statistics.quantile!(vec(abs.(p_whpdiffs)), plevel)

        targ_sse = sum(p_pdiffs.^2)
        wtsum_sse = sum(p_whpdiffs.^2)

        targ_rmse = sqrt(targ_sse / length(p_pdiffs))
        wtsum_rmse = sqrt(wtsum_sse / length(p_whpdiffs))
        tot_rmse = sqrt((targ_sse + wtsum_sse) / (length(p_pdiffs) + length(p_whpdiffs)))

        # println("halt before: $halt")
        halt = false # targ_max < targstop && wtsum_max < whstop
        #println("halt after: $halt")

        totseconds = time() - tstart

        @printf(" %8i %8i %11.5g %12.7g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g \n",
            iter_calc, fcalls, totseconds, objval, targ_rmse,  wtsum_rmse, tot_rmse, targ_max, wtsum_max, targ_ptile, wtsum_ptile)
    end
    # println("targstop and targ_max: ", targstop, "  ", targ_max)
    halt = targ_max < targstop && wtsum_max < whstop
    return halt
end

function show_iter(; iter_calc, fcalls, totseconds, objval, targ_rmse,  wtsum_rmse, tot_rmse, targ_max, wtsum_max, targ_ptile, wtsum_ptile)
    # declare as global any variables that must persist from one call to the next, and may be changed
    global nshown  # init val 0
    # print headings if needed
    nshown += 1
    if nshown ==1 || mod(nshown, 20) == 0
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


function display_poisson(p_pdiffs, wh, whs)
    # note: we do not want this code included in autodifferentiation if it is called from an objective function
    # p_pdiffs is % diffs from targets - i.e., the ojbective function vector objvec

    # declare as global any variables that must persist from one call to the next, and may be changed
    global fcalls  # init val 0
    global bestobjval  # init val Inf
    global iter_calc  # init val 0
    # also global, but not changed: interval, plevel

    # is this a new iteration, and should its results be shown?
    new_iter = false
    objval = sum(p_pdiffs.^2) / length(p_pdiffs)
    if objval < bestobjval || (fcalls<=5 && objval > bestobjval)
        # I want to see the first early iteration after the objval increased
        bestobjval = objval
        iter_calc += 1
        new_iter = true
        use_iter = mod(iter_calc, interval) == 0 || iter_calc in (0, 1)
    end

    if !new_iter return end

    # here is where we would check for halting, if allowed - but it's not
    if !use_iter return end

    # we only get to here if we are going to use (display) this iteration

    # get statistics for targets
    targ_max = maximum(abs.(p_pdiffs))
    targ_ptile = Statistics.quantile!(vec(abs.(p_pdiffs)), plevel)

    # calculate statistics for display
    p_whpdiffs =  (sum(whs, dims=2) - wh) ./ wh * 100.  # vector of % differences of weights from sums of weights
    wtsum_max = maximum(abs.(p_whpdiffs))
    wtsum_ptile = Statistics.quantile!(vec(abs.(p_whpdiffs)), plevel)

    targ_sse = sum(p_pdiffs.^2)
    wtsum_sse = sum(p_whpdiffs.^2)

    targ_rmse = sqrt(targ_sse / length(p_pdiffs))
    wtsum_rmse = sqrt(wtsum_sse / length(p_whpdiffs))
    tot_rmse = sqrt((targ_sse + wtsum_sse) / (length(p_pdiffs) + length(p_whpdiffs)))

    totseconds = time() - tstart

    show_iter(iter_calc=iter_calc, fcalls=fcalls, totseconds=totseconds,
     objval=objval,
     targ_rmse=targ_rmse,  wtsum_rmse=wtsum_rmse, tot_rmse=tot_rmse,
     targ_max=targ_max, wtsum_max=wtsum_max,
     targ_ptile=targ_ptile, wtsum_ptile=wtsum_ptile)

    return
end


