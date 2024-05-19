using NLopt
using OptimizationNLopt

# Define the objective function
function objective_function(x::Vector{Float64})
    return (x[1] - 1.0)^2 + (x[2] - 2.0)^2
end

# Define the callback function
function callback_function(x, grad)
    # Example stopping criteria: stop if close to [1.0, 2.0]
    tolerance = 1e-4
    stopping_criteria_met = all(abs.(x .- [1.0, 2.0]) .< tolerance)
    return stopping_criteria_met
end

# Set up the optimizer
opt = Opt(:LN_BOBYQA, 2)  # The second argument is the dimension of the problem

# Wrap the objective function
objective_function_wrapped(x::Vector{Float64}, grad::Vector{Float64}) = objective_function(x)
opt.min_objective = objective_function_wrapped

# Set the callback function
opt.callback = callback_function

# Set initial guess
initial_guess = [0.0, 0.0]

# Set tolerances and maximum iterations (optional)
opt.xtol_rel = 1e-6
opt.maxeval = 1000

# Run the optimization and handle the result
try
    (minf, minx, ret) = optimize(opt, initial_guess, callback=callback_function)
    if ret == :FORCED_STOP
        println("Optimization stopped by callback.")
    else
        println("Optimization completed.")
    end
    println("Minimum value: $minf")
    println("Optimal point: $minx")
    println("Return code: $ret")
catch e
    println("An error occurred: ", e)
end
