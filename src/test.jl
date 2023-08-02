A = [1. 2. 3.; 4. 5. 6.; 7. 8. 9.; 10. 11. 12.]
x = [12., 13., 14., 15.]
A'x
targ = [310., 370., 422.]

Base.@kwdef mutable struct test
    a::Vector{Float64}
    b::Int = 0
end 
  
# inner constructor should assign every UNDEFEINED element by keyword
function test(a)
    return test(a=a)
end

z = test(targ)

z
z.a
z.b

xmat'wh
targ = [310., 370., 422.]

# -------------------------------------

Base.@kwdef mutable struct rwp
    # required valeus
    wh::Vector{Float64}
    xmat::Matrix{Float64}

    # calculated from required values
    h::Int = 0
    k::Int = 0
    rwtargets_calc::Vector{Float64} = []

    # optional values that may be created by assignment later
    rwtargets::Vector{Float64} = []
    rwtargets_diff::Vector{Float64} = []

    # bounds and constraints
    xlb::Float64 = -Inf
    xub::Float64 = Inf

    # placeholders in case we create scaled versions of the data
    wh_scaled::Vector{Float64} = []
    xmat_scaled::Vector{Float64} = []
    rwtargets_scaled::Vector{Float64} = []
end  
  
function rwp(wh::Vector{Float64}, xmat::Matrix{Float64})
    # check dimensions
    length(wh) == size(xmat, 1) || throw(DimensionMismatch("wh and xmat must have same # of rows"))   

    h = length(wh)
    k = size(xmat)[2]
    rwtargets_calc = xmat'wh

    rwp(wh=wh, xmat=xmat, h=h, k=k, rwtargets_calc=rwtargets_calc)
end

wh = [12., 13., 14., 15.]
xmat = [1. 2. 3.; 4. 5. 6.; 7. 8. 9.; 10. 11. 12.]

z = rwp(wh, xmat)
fieldnames(typeof(z))
z.wh
z.xmat
z.h
z.k
z.rwtargets_calc
z.rwtargets
z.rwtargets_diff
z.xlb
z.xub
z.wh_scaled
z.xmat_scaled

using Random, Distributions

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
    r = rand(Normal(0., 0.1), k)
    rwtargets = rwtargets .* (1 .+ r)

    tp = rwp(wh, xmat)
    tp.rwtargets = rwtargets
    tp.rwtargets_diff = tp.rwtargets_calc .- tp.rwtargets
    return tp
end

z = mtprw(100, 4)
z.wh
z.xmat
z.rwtargets
z.rwtargets_calc
z.rwtargets_diffs


# -----------------------------------------------------------------
Base.@kwdef mutable struct ReweightProblem
    wh::Vector{Float64} # always required
    xmat::Matrix{Float64} # always required
    rwtargets::Vector{Float64} = []  # optional
  
    rwtargets_calc::Vector{Float64} = [] # will be calculated and enforced
    h::Int = 0 # number of households, calculated
    k::Int = 0 # number of characteristics per household, calculated
  
    # changeable
    rwtargets_diff::Vector{Float64} = []
    xlb::Float64 = -Inf
    xub::Float64 = Inf
    
    # placeholders in case we create scaled versions of the data
    wh_scaled::Vector{Float64} = []
    xmat_scaled::Matrix{Float64} = []
    rwtargets_scaled::Vector{Float64} = []
  end  
  
  function ReweightProblem(wh::Vector{Float64}, xmat::Matrix{Float64})
      # check dimensions
      length(wh) == size(xmat, 1) || throw(DimensionMismatch("wh and xmat must have same # of rows"))   
  
      rwtargets_calc = xmat' * wh
      h = length(wh)
      k = size(xmat)[2]
  
      ReweightProblem(wh=wh, xmat=xmat, rwtargets_calc=rwtargets_calc, h=h, k=k)
  end



