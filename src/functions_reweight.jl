
# module BruteForce

# %% opt functions

function objfn_reweight(
    wh, xmat,
    rwtargets;
    whweight=0.5,
    pow=2.0,
    targstop=true, whstop=true,
    display_progress=true)

    # part 1 get measure of difference from targets
    rwtargets_calc = xmat' * wh  
    # targdiffs = (rwtargets_calc .- rwtargets) ./ rwtargets  * 100. # allocates a tiny bit
    targdiffs = (rwtargets_calc .- rwtargets) # ./ 1e6 # allocates a tiny bit
    ss_targdiffs = sum(targdiffs.^pow)
  
    # part 2 - measure of change in weight
    whdiffs = wh ./ wh .- 1.0
    ss_whdiffs = mean(whdiffs .^pow)
  
    # combine the two measures and take a root
    # objval = (ss_targdiffs / length(targdiffs))*(1. - whweight) +
    #         (ss_whdiffs / length(whdiffs))*whweight
    # objval = objval^(1. / pow)
    objval = ss_targdiffs

  # list extra variables on the return so that they are available to the callback function
  return objval # , targdiffs, whdiffs, targstop, whstop
end

