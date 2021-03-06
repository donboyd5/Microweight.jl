
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

function objfn_direct(shares, wh, xmat, geotargets,
  p_mshares, p_whs, p_calctargets, p_pdiffs, p_whpdiffs,
  whweight,
  pow,
  targstop, whstop,
  display_progress=true)

  # part 1
  p_mshares = reshape(shares, length(wh), :) # matrix of shares will be h x s
  p_whs = wh .* p_mshares # this allocates memory
  p_calctargets = p_whs' * xmat
  p_pdiffs = (p_calctargets .- geotargets) ./ geotargets * 100.  # allocates a tiny bit
  ss_pdiffs = sum((p_pdiffs).^pow)

  # part 2 - get sum of squared diffs from zero for wh diffs
  p_whpdiffs = (sum(p_whs, dims=2) .- wh) ./ wh * 100.
  ss_whpdiffs = sum((p_whpdiffs ).^pow)

  #   # part 3 penalty for negative weights
  #   penalty = sum(p_whs .< 0.) * fcalls

  # combine and take a root
  objval = (ss_pdiffs / length(p_pdiffs))*(1. - whweight) +
           (ss_whpdiffs / length(p_whpdiffs))*whweight
  objval = objval^(1. / pow)

  # list extra variables on the return so that they are available to the callback function
  return objval, p_pdiffs, p_whpdiffs, targstop, whstop
end


function objfn_direct_negpen(shares, wh, xmat, geotargets,
  p_mshares, p_whs, p_calctargets, p_pdiffs, p_whpdiffs,
  whweight,
  pow,
  targstop, whstop,
  display_progress=true)

  # part 1
  p_mshares = reshape(shares, length(wh), :) # matrix of shares will be h x s
  p_whs = wh .* p_mshares # this allocates memory
  p_calctargets = p_whs' * xmat
  p_pdiffs = (p_calctargets .- geotargets) ./ geotargets * 100.  # allocates a tiny bit
  ss_pdiffs = sum(p_pdiffs.^pow)
  # ss_pdiffs = sum((p_pdiffs.^pow)^(1.0/pow))

  # part 2 - get sum of squared diffs from zero for wh diffs
  p_whpdiffs = (sum(p_whs, dims=2) .- wh) ./ wh * 100.
  ss_whpdiffs = sum(p_whpdiffs.^pow)
  # ss_whpdiffs = sum((p_whpdiffs.^pow)^(1.0/pow))

  # combine and take a root
  objval1 = (ss_pdiffs / length(p_pdiffs))*(1. - whweight) +
           (ss_whpdiffs / length(p_whpdiffs))*whweight

  objval1 = objval1^(1. / pow)

  # part 3 penalty for negative weights
  # penalty = sum(p_whs .< 0.) * 100.0
  # penalty = sum(p_whs.^2 .- p_whs) * 1000.
  # penalty = maximum(p_whs.^2)
  penalty = sum((min.(p_whs .- 0.0, 0.0).*-100.0)).^2

  objval = objval1 + penalty

  ChainRules.ignore_derivatives() do
    if penalty < 0.0 # && fcalls < 10
      println("penalty: ", penalty)
      # println(Statistics.quantile!(vec(p_whs), [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]))
      # println("objval before penalty: ", objval1)
      # println("objval after penalty", objval)
      # return
     end
    end

  # list extra variables on the return so that they are available to the callback function
  return objval, p_pdiffs, p_whpdiffs, targstop, whstop
end


# function objfn_direct_scaled(shares, wh, xmat, geotargets,
#     p_mshares, p_whs, p_calctargets, p_pdiffs, p_whpdiffs,
#     interval,
#     whweight=.5,
#     pow=4,
#     display_progress=true)

#   # part 1
#   # if fcalls in (2, 7) println("shares: ", shares) end
#   p_mshares = reshape(shares, length(wh), :) # matrix of shares will be h x s
#   p_whs = wh .* p_mshares ./ s_scale # this allocates memory
#   p_calctargets = p_whs' * xmat
#   p_pdiffs = (p_calctargets .- geotargets) ./ geotargets * 100.  # allocates a tiny bit
#   ss_pdiffs = sum((p_pdiffs).^pow)

#   # part 2 - get sum of squared diffs from zero for wh diffs
#   p_whpdiffs = (sum(p_whs, dims=2) .- wh) ./ wh * 100.
#   ss_whpdiffs = sum((p_whpdiffs ).^pow)

#   # combine
#   # objval = ss_pdiffs + ss_whpdiffs*whweight
#   global whweight2
#   # whweight2 = ss_whpdiffs / ss_pdiffs
#   whweight2 = whweight
#   objval = (ss_pdiffs / length(p_pdiffs))*(1. - whweight2) + (ss_whpdiffs / length(p_whpdiffs))*whweight2
#   objval = objval^(1. / pow)

#   if display_progress
#     display_status(interval, geotargets, p_calctargets, wh, p_whs, objval)
#   end

#   return objval
# end

# function objfn_direct_negpen(shares, wh, xmat, geotargets,
#   p_mshares, p_whs, p_calctargets, p_pdiffs, p_whpdiffs,
#   interval, whweight, display_progress=true)

#   # part 1
#   p_mshares = reshape(shares, length(wh), :) # matrix of shares will be h x s
#   p_whs = wh .* p_mshares # this allocates memory
#   p_calctargets = p_whs' * xmat
#   p_pdiffs = (p_calctargets .- geotargets) ./ geotargets * 100.  # allocates a tiny bit
#   ss_pdiffs = sum(p_pdiffs.^2)

#   # part 2 - get sum of squared diffs from zero for wh diffs
#   p_whpdiffs = (sum(p_mshares, dims=2) .- 1.) * 100.
#   ss_whpdiffs = sum(p_whpdiffs.^2)

#   # part 3 penalty for negative weights
#   penalty = sum(p_whs .< 0.) * fcalls

#   # combine
#   objval = ss_pdiffs + ss_whpdiffs*whweight + penalty

#   if display_progress
#      display1(interval, geotargets, p_calctargets, wh, p_whs, objval)
#   end

#   return objval
# end


# end # module