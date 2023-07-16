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
    wh_scaled::Matrix{Float64}
    xmat_scaled::Matrix{Float64}
    geotargets_scaled::Matrix{Float64}

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

##############################################################################
##
## Reweight Problem
##
##############################################################################

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




##############################################################################
##
## Results struct
##
##############################################################################
# mutable struct Result
#   # https://discourse.julialang.org/t/default-value-of-some-fields-in-a-mutable-struct/33408
#   method::Symbol # we must create a Result struct with at least this field
#   minimizer::Vector{Float64}
#   minimum::Float64
#   solver_result
#   function Result(method; minimizer=[Inf], minimum=Inf, solver_result=nothing) =
#     Result(method, minimizer, minimum, solver_result)
# end

@with_kw mutable struct Result
  # https://discourse.julialang.org/t/default-value-of-some-fields-in-a-mutable-struct/33408
  approach::Symbol = :missing
  method::Symbol = :missing
  success::Bool = false
  iterations::Int = -999
  eseconds::Float64 = -Inf
  sspd::Float64 = Inf
  beta::Vector{Float64} = [Inf]
  # beta0::Vector{Float64} = [Inf]
  beta0 = nothing
  shares::Vector{Float64} = [Inf]
  # shares0::Vector{Float64} = [Inf]
  shares0 = nothing
  whs::Array{Float64,2} = Array{Float64}(undef, 0, 0)
  wh_calc = nothing
  wh_pdiffs = nothing
  wh_pdqtiles::Vector{Float64} = [Inf]
  geotargets_calc::Array{Float64,2} = Array{Float64}(undef, 0, 0)
  targ_pdiffs::Array{Float64,2} = Array{Float64}(undef, 0, 0)
  targ_pdqtiles::Vector{Float64} = [Inf]
  solver_result = nothing
  problem = nothing
end
