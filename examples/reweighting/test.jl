# test.jl
# https://github.com/pjssilva/NLPModelsAlgencan.jl
# Algencan is a large scale high performance augmented Lagrangian solver written by Ernesto Birgin and Mario Martínez. 
# It has many special features like being able to use the HSL library to speed up the sparse matrix linear algebra and some smart acceleration strategies.

using SPGBox

f(x) = x[1]^2 + (x[2] - 1)^2
# f (generic function with 1 method)

function g!(g, x)
    g[1] = 2 * x[1]
    g[2] = 2 * (x[2] - 1)
end
# g! (generic function with 1 method)

function my_callback(R::SPGBoxResult)
    if R.nit <= 3
        println("hello")
        return false 
    else
        println(fieldnames(typeof(R)))
        return true # terminate the optimization
    end
end
# my_callback (generic function with 1 method)

x = [10.0, 18.0]
R = spgbox!(f, g!, x; callback = my_callback)
println(fieldnames(typeof(R)))
R.x
R.f
R.gnorm
R.nit


# export MA57_SOURCE=/tmp/hsl_ma57-5.2.0.tar.gz
# U:\home\donboyd5\Documents\julia_projects\hsl
# export MA57_SOURCE=/home/donboyd5/Documents/julia_projects/hsl/coinhsl-2021.05.05.tar.gz
# add NLPModelsAlgencan
# using NLPModelsAlgencan
using Printf
using NLPModels
using CUTEst
using NLPModelsAlgencan

"""
Solve a single CUTEst problem given as a parameter in the command line.
"""
function solve_cutest(pname)
    # solver = AlgencanSolver(epsfeas=1.0e-5, epsopt=1.0e-5,
    #     efstain=3.162278e-03, eostain=3.162278e-08, efacc=3.162278e-3,
    #     eoacc=3.162278e-3,
    #     ITERATIONS_OUTPUT_DETAIL=10, NUMBER_OF_ARRAYS_COMPONENTS_IN_OUTPUT=0)
    nlp = CUTEstModel(pname)
    bench_data = @timed status = algencan(nlp)
    finalize(nlp)
    println("Solver status = ", status)
    c = status.counters.counters
    n_fc, n_ggrad, n_hl, n_hlp = c.neval_obj, c.neval_jac, c.neval_hess, c.neval_hprod
    println("Perfomance data (time, n_fc, n_ggrad, n_hl, n_nlp)")
    @printf("%.4e\t%10d\t%10d\t%10d\t%10d\n", bench_data[2], n_fc, n_ggrad, n_hl, n_hlp)
end

# Calls a simple problem to compile the Julia code in order to get a reasonable
# timing information at the end.
set_mastsif()
solve_cutest("HS110")
println("\n\n", '*'^40, " Solving ", ARGS[1], "\n")
solve_cutest(ARGS[1])

using JuMP
using NLPModels, NLPModelsJuMP, NLPModelsAlgencan


println("Building the first model...\n")
m = Model()
@variable(m, 3.0 <= var[1:2] <= 5.0)
@NLobjective(m, Min, 123.45*(var[1] - π)^2 + sin(var[1]) + (var[2] - ℯ)^2 + sin(π - var[2]))
println(m)

println("Create the bridge to NLPModels..\n")
nlp = MathOptNLPModel(m)

println("Solving...")
status = algencan(nlp, epsfeas=1.0e-5, epsopt=1.0e-5)
println("Solution status: $status.")
print("(Primal) Solution to first problem: ")
println(status.solution, "\n\n")

println("Building second model...\n")
m2 = Model()
@variable(m2, 2.0 <= var2[1:3] <= 5.0)
@NLobjective(m2, Min, (var2[1] - 4)^2 + (var2[2] - 4)^2 + var2[3])
@constraint(m2, -5 <= -(var2[1] + var2[2]))
println(m2)

println("Create the bridge to NLPModels..\n")
cnlp = MathOptNLPModel(m2)

println("Solving...")
status = algencan(cnlp, epsfeas=1.0e-8, epsopt=1.0e-6)

println("Solution status: $status.")
print("Primal solution to second problem: ")
println(status.solution)
print("Multipliers: ")
println(status.multipliers)
println("Full solve time = ", status.elapsed_time, "s")





