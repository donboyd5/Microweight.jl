# goodvals = (:method, :s)
# keep = filter(p -> first(p) in goodvals, z)
# zz = f(10, method="y"; keep...)

function clean_kwargs(kwargs_passed, kwkeys_allowed)
    # returns a Dict that can be used as kwargs...
    # kwargs_passed::pairs(::NamedTuple), kwargs_allowed::pairs(::NamedTuple)
    filter(p -> first(p) in kwkeys_allowed, kwargs_passed)
end


# function display_progress(fcalls, interval, p_pdiffs, p_whpdiffs, ss_pdiffs, ss_whpdiffs, sspd)
#     Zygote.ignore() do
#         if mod(fcalls, interval) == 0 || fcalls ==1
#             maxabstarg = maximum(abs.(p_pdiffs))
#             maxabswt = maximum(abs.(p_whpdiffs))
#             # nshown - how many lines have we displayed, including this one?
#             if fcalls == 1
#                 nshown = 1
#             else
#                 nshown = fcalls / interval + 1
#             end
#             if nshown ==1 || mod(nshown, 20) == 0
#                 println()
#                 println("  fcalls     ss_targets  ss_weightsums    ss_combined     maxabstarg       maxabswt   nshown")
#             end
#             @printf("%8i %14.5g %14.5g %14.5g %14.5g %14.5g %9.4g \n", fcalls, ss_pdiffs, ss_whpdiffs, sspd, maxabstarg, maxabswt, nshown)
#         end
#     end
# end


# function display1(interval, geotargets, p_calctargets, wh, p_whs, objval=nothing)
#     # , p_pdiffs, p_whpdiffs, ss_pdiffs, ss_whpdiffs, sspd
#     global fcalls
#     fcalls += 1

#     global tstart
#     Zygote.ignore() do
#         if mod(fcalls, interval) == 0 || fcalls ==1
#             # maxabswt = maximum(abs.(p_whpdiffs))
#             # nshown - how many lines have we displayed, including this one?
#             # this if-else block allows us to count sequentially without keeping another global variable
#             if interval == 1 || fcalls == 1
#                 nshown = fcalls
#             else
#                 nshown = fcalls / interval + 1
#             end

#             if nshown ==1 || mod(nshown, 20) == 0
#                 println()
#                 println("  fcalls     ss_targets  ss_weightsums         objval     maxabstarg       maxabswt   nshown  totseconds    whweight     s_scale")
#             end
#             p_pdiffs = (p_calctargets .- geotargets) ./ geotargets * 100.
#             # ss_pdiffs = sum(p_pdiffs.^2)
#             # ss_pdiffs = (sum(p_pdiffs.^pow) / length(p_pdiffs))^(1 / 4)
#             ss_pdiffs = Statistics.quantile!(vec(abs.(p_pdiffs)), plevel)
#             maxabstarg = maximum(abs.(p_pdiffs))

#             p_whpdiffs = (sum(p_whs, dims=2) .- wh) ./ wh * 100.
#             # ss_whpdiffs = sum(p_whpdiffs.^2)
#             # ss_whpdiffs = (sum(p_whpdiffs.^pow) / length(p_whpdiffs))^(1 / pow)
#             ss_whpdiffs = Statistics.quantile!(vec(abs.(p_whpdiffs)), plevel)
#             maxabswt = maximum(abs.(p_whpdiffs))

#             if objval === nothing
#                 objval = ss_pdiffs
#             end

#             totseconds = time() - tstart

#             #@printf("%8i %14.5g %14.5g %8.4g \n", fcalls, ss_pdiffs, maxabstarg, nshown)
#             @printf("%8i %14.5g %14.5g %14.5g %14.5g %14.5g %8.4g %11.5g %11.5g %11.5g \n", fcalls, ss_pdiffs, ss_whpdiffs, objval, maxabstarg, maxabswt, nshown, totseconds, whweight2, s_scale)
#             # @printf("%8i %14.5g %14.5g %14.5g %14.5g %14.5g %8.4g \n", fcalls, ss_pdiffs, ss_whpdiffs, sspd, maxabstarg, maxabswt, nshown)
#         end
#     end
# end


# function display2(interval, geotargets, p_calctargets, wh, p_whs, objval=nothing)
#     global fcalls
#     fcalls += 1

#     # global tstart
#     Zygote.ignore() do
#         if mod(fcalls, interval) == 0 || fcalls ==1
#             # nshown - how many lines have we displayed, including this one?
#             # this if-else block allows us to count sequentially without keeping another global variable
#             if interval == 1 || fcalls == 1
#                 nshown = fcalls
#             else
#                 nshown = fcalls / interval + 1
#             end

#             if nshown ==1 || mod(nshown, 20) == 0
#                 println()
#                 hdr1 = "  nshown   fcalls  totseconds       objval    targ_rmse   wtsum_rmse     tot_rmse     targ_max    wtsum_max"
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

#             if objval === nothing
#                 objval = ss_pdiffs
#             end

#             totseconds = time() - tstart

#             @printf("%8i %8i %11.5g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g \n",
#               nshown, fcalls, totseconds, objval, targ_rmse,  wtsum_rmse, tot_rmse, targ_max, wtsum_max, targ_ptile, wtsum_ptile)
#         end
#     end
# end



function display_status(interval, geotargets, p_calctargets, wh, p_whs, objval=nothing)
    global fcalls  # init val 0
    global nshown  # init val 0
    global bestobjval  # init val Inf
    global iter_calc  # init val 0

    # global tstart
    Zygote.ignore() do
        fcalls += 1
        new_iter = false

        if objval < bestobjval || iter_calc in (0, 1)
            bestobjval = objval
            new_iter = true
            iter_calc += 1
        end

        show_iter = mod(iter_calc, interval) == 0 || iter_calc in (0, 1)
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
            p_pdiffs = (p_calctargets .- geotargets) ./ geotargets * 100.
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

            totseconds = time() - tstart

            @printf(" %8i %8i %11.5g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g \n",
              iter_calc, fcalls, totseconds, objval, targ_rmse,  wtsum_rmse, tot_rmse, targ_max, wtsum_max, targ_ptile, wtsum_ptile)
        end
    end
end


function cb_direct(shares, objval, p_pdiffs, p_whpdiffs, interval, targstop, whstop)
    halt = false

    # declare as global any variables that must persist from one call to the next, and may be changed
    global fcalls  # init val 0
    global nshown  # init val 0
    global bestobjval  # init val Inf
    global iter_calc  # init val 0

    fcalls += 1
    new_iter = false

    if objval < bestobjval || (fcalls<=5 && objval > bestobjval)
        bestobjval = objval
        new_iter = true
        iter_calc += 1
    end

        # if objval < bestobjval || iter_calc in (0, 1)
        #     bestobjval = objval
        #     new_iter = true
        #     iter_calc += 1
        # end

    show_iter = mod(iter_calc, interval) == 0 || iter_calc in (0, 1)
    # show_iter = true

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
        wtsum_max = maximum(abs.(p_whpdiffs))
        wtsum_ptile = Statistics.quantile!(vec(abs.(p_whpdiffs)), plevel)

        targ_sse = sum(p_pdiffs.^2)
        wtsum_sse = sum(p_whpdiffs.^2)

        targ_rmse = sqrt(targ_sse / length(p_pdiffs))
        wtsum_rmse = sqrt(wtsum_sse / length(p_whpdiffs))
        tot_rmse = sqrt((targ_sse + wtsum_sse) / (length(p_pdiffs) + length(p_whpdiffs)))

        halt = targ_max < targstop && wtsum_max < whstop

        totseconds = time() - tstart

        @printf(" %8i %8i %11.5g %12.7g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g \n",
            iter_calc, fcalls, totseconds, objval, targ_rmse,  wtsum_rmse, tot_rmse, targ_max, wtsum_max, targ_ptile, wtsum_ptile)
    end

    return halt
end

# cb = tr -> begin
#             push!(xs, tr[end].metadata["x"])
#             false
#         end

function cb1(p, l, ss_pdiffs)
    # println(fieldnames(typeof(tr)))
    println("hello")
    println("loss: $l")
    println("sum pdiffs ", ss_pdiffs)
    return false
end


function cb2(p, l, p_pdiffs, p_whpdiffs)
    # println(fieldnames(typeof(tr)))
    println("hello")
    println("loss: $l")
    println("sum pdiffs ", p_pdiffs[1], sum(p_whpdiffs))
    return false
end


function display_status2(interval, geotargets, p_calctargets, wh, p_whs, objval=nothing)
    global fcalls  # init val 0
    global nshown  # init val 0
    global bestobjval  # init val Inf
    global iter_calc  # init val 0

    # global tstart
    Zygote.ignore() do
        fcalls += 1
        new_iter = false

        if objval < bestobjval || (fcalls<=5 && objval > bestobjval)
            bestobjval = objval
            new_iter = true
            iter_calc += 1
        end

        # if objval < bestobjval || iter_calc in (0, 1)
        #     bestobjval = objval
        #     new_iter = true
        #     iter_calc += 1
        # end

        show_iter = mod(iter_calc, interval) == 0 || iter_calc in (0, 1)
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
            p_pdiffs = (p_calctargets .- geotargets) ./ geotargets * 100.
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

            totseconds = time() - tstart

            @printf(" %8i %8i %11.5g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g %12.5g \n",
              iter_calc, fcalls, totseconds, objval, targ_rmse,  wtsum_rmse, tot_rmse, targ_max, wtsum_max, targ_ptile, wtsum_ptile)
        end
    end
end


