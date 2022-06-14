# goodvals = (:method, :s)
# keep = filter(p -> first(p) in goodvals, z)
# zz = f(10, method="y"; keep...)

function clean_kwargs(kwargs_passed, kwkeys_allowed)
    # returns a Dict that can be used as kwargs...
    # kwargs_passed::pairs(::NamedTuple), kwargs_allowed::pairs(::NamedTuple)
    filter(p -> first(p) in kwkeys_allowed, kwargs_passed)
end