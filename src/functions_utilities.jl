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
            @printf("%8i %14.5g %14.5g %14.5g %14.5g %14.5g %8.4g \n", fcalls, ss_pdiffs, ss_whpdiffs, sspd, maxabstarg, maxabswt, nshown)
        end
    end
end


function display1(fcalls, interval, geotargets, p_calctargets, wh, p_whs)
    # , p_pdiffs, p_whpdiffs, ss_pdiffs, ss_whpdiffs, sspd
    global fcalls
    fcalls += 1
    Zygote.ignore() do
        if mod(fcalls, interval) == 0 || fcalls ==1
            #
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
                println("  fcalls     ss_targets  ss_weightsums         objval     maxabstarg       maxabswt   nshown")
            end
            p_pdiffs = (p_calctargets .- geotargets) ./ geotargets * 100.
            ss_pdiffs = sum(p_pdiffs.^2)
            maxabstarg = maximum(abs.(p_pdiffs))

            p_whpdiffs = (sum(p_whs, dims=2) .- wh) ./ wh * 100.
            ss_whpdiffs = sum(p_whpdiffs.^2)
            maxabswt = maximum(abs.(p_whpdiffs))

            #@printf("%8i %14.5g %14.5g %8.4g \n", fcalls, ss_pdiffs, maxabstarg, nshown)
            @printf("%8i %14.5g %14.5g %14.5g %14.5g %14.5g %8.4g \n", fcalls, ss_pdiffs, ss_whpdiffs, ss_pdiffs, maxabstarg, maxabswt, nshown)
            # @printf("%8i %14.5g %14.5g %14.5g %14.5g %14.5g %8.4g \n", fcalls, ss_pdiffs, ss_whpdiffs, sspd, maxabstarg, maxabswt, nshown)
        end
    end
end

function changeit(fcalls)
    fcalls += 7
end