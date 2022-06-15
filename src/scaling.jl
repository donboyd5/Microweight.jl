
#=

See http://www2.imm.dtu.dk/pubdb/edoc/imm3215.pdf
Example 3.19

# noted here: http://carlmeyer.com/Least%20Squares%20Discussion.pdf

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


function scale_prob(prob; scaling, scaling_target_goal)
    if scaling
        prob.wh_scaled = prob.wh
        # targscale = targ_goal ./ maximum(abs.(tp.geotargets), dims=1)
        # targscale = targ_goal ./ sum(tp.geotargets, dims=1)
        scaling_target_factor = scaling_target_goal ./ sum(prob.xmat, dims=1)
        prob.geotargets_scaled = scaling_target_factor .* prob.geotargets
        prob.xmat_scaled = scaling_target_factor .* prob.xmat
    else
        prob.wh_scaled = prob.wh
        prob.geotargets_scaled = prob.geotargets
        prob.xmat_scaled = prob.xmat
    end
    return prob
end
