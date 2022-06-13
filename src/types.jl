##############################################################################
##
## Geoweight Problem
##
##############################################################################

mutable struct GeoweightProblem
    wh::Matrix{Float64}
    xmat::Matrix{Float64}
    geotargets::Matrix{Float64}
    h::Int  # number of households
    k::Int # number of characteristics per household
    s::Int  # number of states or areas
    target_sums::Matrix{Float64}
    target_calcs
    target_diffs

    function GeoweightProblem(wh, xmat, geotargets)
        # check dimensions
        length(wh) == size(xmat, 1) || throw(DimensionMismatch("wh and xmat must have same # of rows"))
        size(xmat, 2) == size(geotargets, 2) || throw(DimensionMismatch("xmat and geotargets must have same # of columns"))

        # check approximate equality of calculated national targets and provided national targets (sum of states)
        # https://discourse.julialang.org/t/approximate-equality/8952
        target_sums = sum(geotargets, dims=1)
        target_calcs = sum(wh .* xmat, dims=1)
        target_diffs = target_calcs - target_sums
        isapprox(target_calcs, target_sums) ||
          @warn "Sums of state targets are not approximately equal to calculated amounts:sum(wh .* xmat, dims=1). Inspect inputs carefully." target_sums, target_calcs, target_diffs

        h = length(wh)
        k = size(xmat)[2]
        s = size(geotargets)[1]
        new(wh, xmat, geotargets, h, k, s, target_sums, target_calcs, target_diffs)
    end
end