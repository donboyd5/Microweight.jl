function scale_prob(geoproblem, target_goal=10.0)
    # xmat scaling multiplicative based on max abs of each col of geotargets
    target_goal = 10.0
    # targscale = targ_goal ./ maximum(abs.(tp.geotargets), dims=1)
    # targscale = targ_goal ./ sum(tp.geotargets, dims=1)
    geoproblem.scaled = true
    geoproblem.target_scale = target_goal ./ sum(geoproblem.xmat, dims=1)
    geoproblem.geotargets_scaled = geoproblem.target_scale .* geoproblem.geotargets
    geoproblem.xmat_scaled = geoproblem.target_scale .* geoproblem.xmat
    return geoproblem
end
