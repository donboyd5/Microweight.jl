
function clean_kwargs(kwargs_passed, kwkeys_allowed)
    # returns a Dict that can be used as kwargs...
    # kwargs_passed::pairs(::NamedTuple), kwargs_allowed::pairs(::NamedTuple)
    filter(p -> first(p) in kwkeys_allowed, kwargs_passed)
end



function kwargs_keep(kwargs; kwkeys_method=nothing, kwkeys_algo=nothing, kwargs_defaults=nothing)
    if isnothing(kwkeys_method) && isnothing(kwkeys_algo) &&  isnothing(kwargs_defaults) return nothing end

    # merge allowable sets of keys - these are named tuples
    kwkeys_allowed = kwkeys_method
    if !isnothing(kwkeys_algo) kwkeys_allowed = (kwkeys_method..., kwkeys_algo...) end
    println("kwargs allowed: ", kwkeys_allowed)

    println("kwargs requested: ", kwargs)

    # create a Dict from allowable kwargs...
    if !isnothing(kwkeys_allowed)
      kwargs_keep = filter(p -> first(p) in kwkeys_allowed, kwargs)
    else
      kwargs_keep = nothing
    end
    println("allowable requested kwargs:\n $kwargs_keep")

    # merge defaults, if any, with kwargs - second will overwrite first
    if !isnothing(kwargs_defaults)
      println("kwargs defaults: ", kwargs_defaults)
      kwargs_keep = merge(kwargs_defaults, kwargs_keep) # overwrite any defaults with requested values
    else
      println("no kwargs defaults")
    end
    println("kwargs passed, (conflicting requested values will override defaults):\n $kwargs_keep")

    return kwargs_keep
  end

