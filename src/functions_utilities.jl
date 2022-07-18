
function clean_kwargs(kwargs_passed, kwkeys_allowed)
    # returns a Dict that can be used as kwargs...
    # kwargs_passed::pairs(::NamedTuple), kwargs_allowed::pairs(::NamedTuple)
    filter(p -> first(p) in kwkeys_allowed, kwargs_passed)
end



function kwargs_keep(kwargs; kwkeys_method=NamedTuple(), kwkeys_algo=NamedTuple(), kwargs_defaults=Dict())
    println()
    println("default kwargs (if any):\n $kwargs_defaults")
    kwkeys_allowed = (kwkeys_method..., kwkeys_algo...)
    println("kwargs allowed: ", kwkeys_allowed)

    println("kwargs requested: ", kwargs)

    # create a Dict from allowable kwargs...
    kwargs_keep = filter(p -> first(p) in kwkeys_allowed, kwargs)
    println("allowable requested kwargs:\n $kwargs_keep")

    # merge defaults, if any, with kwargs - second will overwrite first
    kwargs_keep = merge(kwargs_defaults, kwargs_keep) # overwrite any defaults with requested values
    println("kwargs passed (requested values override defaults when conflicts arise):\n $kwargs_keep")
    println()

    return kwargs_keep
  end
