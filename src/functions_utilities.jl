# goodvals = (:method, :s)
# keep = filter(p -> first(p) in goodvals, z)
# zz = f(10, method="y"; keep...)

function clean_kwargs(kwargs_passed, kwkeys_allowed)
    # returns a Dict that can be used as kwargs...
    # kwargs_passed::pairs(::NamedTuple), kwargs_allowed::pairs(::NamedTuple)
    filter(p -> first(p) in kwkeys_allowed, kwargs_passed)
end


function display_progress(fcalls, interval, p_pdiffs, p_whpdiffs, ss_pdiffs, ss_whpdiffs, sspd)
    Zygote.ignore() do
        if mod(fcalls, interval) == 0 || fcalls ==1
            maxabstarg = maximum(abs.(p_pdiffs))
            maxabswt = maximum(abs.(p_whpdiffs))
            # nshown - how many lines have we displayed, including this one?
            if fcalls == 1
                nshown = 1
            else
                nshown = fcalls / interval + 1
            end
            if nshown ==1 || mod(nshown, 20) == 0
                println()
                println("  fcalls     ss_targets  ss_weightsums    ss_combined     maxabstarg       maxabswt   nshown")
            end
            @printf("%8i %14.5g %14.5g %14.5g %14.5g %14.5g %9.4g \n", fcalls, ss_pdiffs, ss_whpdiffs, sspd, maxabstarg, maxabswt, nshown)
        end
    end
end


function display1(interval, geotargets, p_calctargets, wh, p_whs, objval=nothing)
    # , p_pdiffs, p_whpdiffs, ss_pdiffs, ss_whpdiffs, sspd
    global fcalls
    fcalls += 1

    global tstart
    Zygote.ignore() do
        if mod(fcalls, interval) == 0 || fcalls ==1
            # maxabswt = maximum(abs.(p_whpdiffs))
            # nshown - how many lines have we displayed, including this one?
            # this if-else block allows us to count sequentially without keeping another global variable
            if interval == 1 || fcalls == 1
                nshown = fcalls
            else
                nshown = fcalls / interval + 1
            end

            if nshown ==1 || mod(nshown, 20) == 0
                println()
                println("  fcalls     ss_targets  ss_weightsums         objval     maxabstarg       maxabswt   nshown  totseconds    whweight     s_scale")
            end
            p_pdiffs = (p_calctargets .- geotargets) ./ geotargets * 100.
            # ss_pdiffs = sum(p_pdiffs.^2)
            # ss_pdiffs = (sum(p_pdiffs.^pow) / length(p_pdiffs))^(1 / 4)
            ss_pdiffs = Statistics.quantile!(vec(abs.(p_pdiffs)), plevel)
            maxabstarg = maximum(abs.(p_pdiffs))

            p_whpdiffs = (sum(p_whs, dims=2) .- wh) ./ wh * 100.
            # ss_whpdiffs = sum(p_whpdiffs.^2)
            # ss_whpdiffs = (sum(p_whpdiffs.^pow) / length(p_whpdiffs))^(1 / pow)
            ss_whpdiffs = Statistics.quantile!(vec(abs.(p_whpdiffs)), plevel)
            maxabswt = maximum(abs.(p_whpdiffs))

            if objval === nothing
                objval = ss_pdiffs
            end

            totseconds = time() - tstart

            #@printf("%8i %14.5g %14.5g %8.4g \n", fcalls, ss_pdiffs, maxabstarg, nshown)
            @printf("%8i %14.5g %14.5g %14.5g %14.5g %14.5g %8.4g %11.5g %11.5g %11.5g \n", fcalls, ss_pdiffs, ss_whpdiffs, objval, maxabstarg, maxabswt, nshown, totseconds, whweight2, s_scale)
            # @printf("%8i %14.5g %14.5g %14.5g %14.5g %14.5g %8.4g \n", fcalls, ss_pdiffs, ss_whpdiffs, sspd, maxabstarg, maxabswt, nshown)
        end
    end
end

function changeit(fcalls)
    fcalls += 7
end