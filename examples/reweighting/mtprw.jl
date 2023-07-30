using Distributions, Random


mutable struct ReweightProblem
    wh::Vector{Float64}
    xmat::Matrix{Float64}
    rwtargets::Vector{Float64}
    h::Int  # number of households
    k::Int # number of characteristics per household
    rwtargets_calc::Vector{Float64}
    rwtargets_diff::Vector{Float64}
  
    # placeholders in case we create scaled versions of the data
    wh_scaled::Vector{Float64}
    xmat_scaled::Matrix{Float64}
    rwtargets_scaled::Vector{Float64}
  
    function ReweightProblem(wh, xmat, rwtargets)
        # check dimensions
        length(wh) == size(xmat, 1) || throw(DimensionMismatch("wh and xmat must have same # of rows"))
        size(xmat, 2) == length(rwtargets) || throw(DimensionMismatch("xmat # of columns must equal length of reweight_targets"))
  
        rwtargets_calc = xmat' * wh
        rwtargets_diff = rwtargets_calc - rwtargets
  
        h = length(wh)
        k = size(xmat)[2]
        new(wh, xmat, rwtargets, h, k, rwtargets_calc, rwtargets_diff)
    end
  end


function mtprw(h, k; pctzero=0.0)
    Random.seed!(123)
    xsd=.005
    wsd=.005
    # pctzero=0.10
    # h = 8
    # k = 2

    # create xmat
    d = Normal(0., xsd)
    r = rand(d, (h, k)) # xmat dimensions
    xmat = 100 .+ 20 .* r

    # set random indexes to zero
    nzeros = round(Int, pctzero * length(xmat))
    # Generate a random set of indices to set to zero
    zindices = randperm(length(xmat))[1:nzeros]
    # Set the elements at the selected indices to zero
    xmat[zindices] .= 0.0 # this works even when pctzero is 0.0

    # create wh
    d = Normal(0., wsd)
    r = rand(d, h) # wh dimensions
    wh = 10 .+ 10 .* (1 .+ r)

    # calc  sums and add noise to get targets
    rwtargets = xmat' * wh
    r = rand(Normal(0., 0.05), k)
    rwtargets = rwtargets .* (1 .+ r)

    return ReweightProblem(wh, xmat, rwtargets)
end


# tp = mtprw(100, 4, pctzero=.3)
# tp = mtprw(1000, 10, pctzero=.3)
# tp = mtprw(10_000, 40, pctzero=.3)
