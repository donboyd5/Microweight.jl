
# module BruteForce

# %% basic functions
function fwhs(shares, wh) # , xmat
    # matrix of shares will be h x s
    mshares = reshape(shares, length(wh), :)
    whs = wh .* mshares
    whs
  end

#   function fgeotargets(shares, wh, xmat)
#     # matrix of shares will be h x s
#     whs = fwhs(shares, wh, xmat)
#     whs' * xmat
#   end

#   function targdiffs(shares, wh, xmat, geotargets)
#     # matrix of shares will be h x s
#     calctargets = fgeotargets(shares, wh, xmat)
#     calctargets .- geotargets
#   end

#   function targpdiffs(shares, wh, xmat, geotargets)
#     # matrix of shares will be h x s
#     diffs = targdiffs(shares, wh, xmat, geotargets)
#     diffs ./ geotargets * 100.
#   end


# %% opt functions
function fcons(shares, wh, xmat, geotargets, p_mshares, p_whs, p_calctargets, p_pdiffs, p_whpdiffs, interval, targweight, display_progress=true)

    # part 1
    p_mshares = reshape(shares, length(wh), :) # matrix of shares will be h x s
    p_whs = wh .* p_mshares # this allocates memory
    p_calctargets = p_whs' * xmat
    p_pdiffs = (p_calctargets .- geotargets) ./ geotargets * 100.  # allocates a tiny bit
    ss_pdiffs = sum(p_pdiffs.^2)

    # part 2 - get sum of squared diffs from zero for wh diffs
    p_whpdiffs = (sum(p_mshares, dims=2) .- 1.) * 100.
    ss_whpdiffs = sum(p_whpdiffs.^2)

    # combine
    sspd = ss_pdiffs*targweight + ss_whpdiffs*(1. - targweight)

        # display_progress = false
    if display_progress
       display1(fcalls, interval, geotargets, p_calctargets, wh, p_whs)
    end

    # report
    # Zygote.ignore() do
    #     if mod(fcalls, interval) == 0 || fcalls ==1
    #         maxabstarg = maximum(abs.(p_pdiffs))
    #         maxabswt = maximum(abs.(p_whpdiffs))
    #         # nshown - how many lines have we displayed, including this one?
    #         if fcalls == 1
    #             nshown = 1
    #         else
    #             nshown = fcalls / interval + 1
    #         end
    #         if nshown ==1 || mod(nshown, 20) == 0
    #             println()
    #             println("  fcalls     ss_targets  ss_weightsums    ss_combined     maxabstarg       maxabswt   nshown")
    #         end
    #         @printf("%8i %14.5g %14.5g %14.5g %14.5g %14.5g %8.4g \n", fcalls, ss_pdiffs, ss_whpdiffs, sspd, maxabstarg, maxabswt, nshown)
    #     end
    # end
    return sspd
end

# end # module