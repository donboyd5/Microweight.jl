
using Optimization
using OptimizationOptimJL
import .check as ck

Dict3 = Dict(:a => 1, :b => "one")
Dict3[:b]

cb = tr -> begin
            # push!(xs, tr[end].metadata["x"])
            println(tr)
            false
        end
cb("don")

module check
    # You don’t have Classes in Julia. Instead you first define the data
    # structure (struct, e.g. what kind of data your objects should hold) and
    # what type of data the structure will hold. Afterwards you create instances
    # of those “objects” with concrete values for the variables (could also be
    # variables from somewhere else).
    using Random

    mutable struct workstore
        # struct for working storage
        h::Int
        k::Int
        s::Int
        mshares::Array{Float64,2} # h, 1
        whs::Array{Float64,2} # h, s
        calctargets::Array{Float64,2} # s, k
        pdiffs::Array{Float64,2} # s, k

        mult::Array{Float64, 1}  # temp
        x::Array{Float64, 1}  # temp
        xmat::Array{Float64,2}

        function workstore(h, k, s)
            # constructor
            Random.seed!(123)
            mshares = zeros(h, s)
            whs = zeros(h, s)
            calctargets = zeros(s, k)
            pdiffs = zeros(s, k)

            x = randn(k)
            mult = randn(k)
            xmat = zeros(h, k)
            new(h, k, s, mshares, whs, calctargets, pdiffs, x, mult, xmat)
        end
  end

end


# %% memory allocation

function mem_not_pre(xmat)
  newmat = xmat .^ 2
  return newmat
end
@btime mem_not_pre(tp.xmat)
#  320.958 ns (1 allocation: 3.25 KiB)
# 181.300 μs (2 allocations: 1.53 MiB)

function mem_pre(p, xmat)
    p.xmat = xmat .^ 2
    return p.xmat
end
@btime mem_pre(a, tp.xmat)
# 271.588 ns (1 allocation: 3.25 KiB)
# 179.600 μs (2 allocations: 1.53 MiB)

function mem_pre2(p, xmat)
    p.xmat = xmat .^ 2
    return # no return statement
end
@btime mem_pre2(a, tp.xmat)
#  262.105 ns (1 allocation: 3.25 KiB)
# 104.300 μs (0 allocations: 0 bytes)

function mem_pre3(p, xmat)
    for i in 1:size(xmat)[1]
        for j in 1:size(xmat)[2]
            p.xmat[i, j] = xmat[i, j] ^ 2
        end
    end
    return p.xmat
end
@btime mem_pre3(a, tp.xmat)
# 267.347 ns (0 allocations: 0 bytes)
# 104.900 μs (0 allocations: 0 bytes)

function testit_no()
    for i in 1:100000
        res = mem_not_pre(tp.xmat)
    end
    return res
end

function testit_yes()
    for i in 1:100000
        res = mem_pre2(a, tp.xmat)
    end
    return res
end

function testit_yes3()
    for i in 1:100000
        res = mem_pre3(a, tp.xmat)
    end
    return res
end


@btime testit_no()
# 2.896 s (20000 allocations: 14.90 GiB)
# 28.923 s (200000 allocations: 149.02 GiB)

@btime testit_yes()
# 1.114 s (0 allocations: 0 bytes)
# 11.263 s (0 allocations: 0 bytes)

@btime testit_yes3()
# 1.176 s (0 allocations: 0 bytes)
# 11.191 s (0 allocations: 0 bytes)



tp = mtp(100, 8, 4)
tp = mtp(1000, 20, 8)
tp = mtp(10000, 50, 20)
tp = mtp(30000, 50, 20)


a = ck.workstore(tp.h, tp.k, tp.s)
a.xmat

@btime  mem_not_pre(tp.xmat)

@btime testit_no()
# 285.700 μs (1000 allocations: 3.17 MiB)




tp.h






# %% optimization test -- works
function caller2(h, k, s)
    p = ck.workstore(h, k, s)
    f(x) = sum(x.^2 + x.^-2) .+ sum(p.mult)

    fp = (x, p) -> f(x)
    x0 = ones(size(p.x)) ./ 2.
    println(fp(x0, p))

    prob = OptimizationProblem(fp, x0, p)
    sol = Optimization.solve(prob, Optim.NelderMead())
    println(fp(sol.u, p))
    sol
end

sol = caller2(8, 3, 4)

p = ck.workstore(h, k, s)
# f10(x) = sum(p.mult .* x.^2 + x.^-4)
f10(x) = sum(x.^2 + x.^-2) .+ sum(p.mult)
fp10 = (x, p) -> f10(x)
fp10
p.x
p.mult
sum(p.mult)
sum(p.x.^2 + p.x.^-2) + sum(p.mult)
fp10(zeros(3), p)

zeros(3).^-2

sol.retcode
sol.minimum
sol.u
sol.original


res.mult

sum(res.mult .* res.x.^2 + res.x.^-4)

x0 = [1., 2., 3.]
ck.f(x0)
# sol = caller2(x0)

prob = OptimizationProblem(ck.f, x0,p)
sol = Optimization.solve(prob, Optim.NelderMead())


a = ck.workstore(10, 3, 4)
a.h
a.k
a.s
a.mult
 a.mshares
a.whs
a.calctargets
a.pdiffs


rosenbrock(x,p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0,100.0]

prob = OptimizationProblem(rosenbrock,x0,p)
sol = Optimization.solve(prob, Optim.NelderMead())