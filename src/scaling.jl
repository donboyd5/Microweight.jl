
#=
different opinions:

https://scicomp.stackexchange.com/questions/7954/scaling-of-optimisation-function-in-non-linear-least-squares-problem
 for convex least-squares problems, it is helpful to scale a problem so that the
 linear solves used in algorithms are well-conditioned; many solvers already do
 this automatically.

every reasonable optimization algorithm should produce exactly the same sequence
of intermediate points (iterates) whether you scale the objective function or
not.

if Nelder-Mead is consistent but the gradient-based methods aren't, it might be
that the gradient is badly scaled. Eg. a unit change in parameter n affects the
objective function very differently than a unit change in parameter m. Try
scaling the parameters (what's f btw?) to (approximately) the same range, or if
you can even better so that if you change any single parameter by a given amount
the objective function will change similarly
 (eg. http://www.alglib.net/optimization/scaling.php)

 

=#


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
