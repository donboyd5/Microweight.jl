

f2(beta) = sum(beta.^4 .- beta.^2)

beta0 = [0., 1., 2., 3.]
f2(beta0)

using Optimization

f = OptimizationFunction(f2, Optimization.AutoForwardDiff())
fprob = OptimizationProblem(f, beta0)

opt = solve(fprob, Optim.BFGS())

rosenbrock(x,p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2

struct rose
    p1::Float64
    p2::Float64
end

p = rose(1.0, 100.0)

p.p1
rosenbrock(x,p) =  (p.p1 - x[1])^2 + p.p2 * (x[2] - x[1]^2)^2

x0 = zeros(2)
p  = [1.0,100.0]
rosenbrock(x0, p)
f = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = OptimizationProblem(f, x0, p)
sol = solve(prob, Optim.BFGS())

rose1(x, p=nothing) =  (- x[1])^2 + (x[2] - x[1]^2)^2
f = OptimizationFunction(rose1, Optimization.AutoForwardDiff())
prob = OptimizationProblem(f, x0, p)
sol = solve(prob, Optim.BFGS())


y = 7.
z = 3.
x0 = [1., 2., 3.]
f3(x0, y, z)
f3(x, y, z) = sum(x.^4 .- x.^2 .+ y .+ z)
f3c = (x, p) -> f3(x, y, z)
f = OptimizationFunction(f3c, Optimization.AutoForwardDiff())
prob = OptimizationProblem(f, x0)
sol = solve(prob, Optim.BFGS(), show_trace=true)
sol.minimizer
f3c(sol.minimizer, nothing)
f3(sol.minimizer, y, z)


struct BufferedContrivedExample{T}
    buffer::T
end

BufferedContrivedExample(T, N::Integer) = BufferedContrivedExample(Vector{T}(undef, N))

function (bce::BufferedContrivedExample)(a)
    b = bce.buffer
    for i in axes(b, 1)
        b[i] += a + i
    end
    b .* 100
end

bce = BufferedContrivedExample(Float64, 100)
bce2 = BufferedContrivedExample(Int64, 10)

bce(1.0)
fieldnames(typeof(bce))
bce = nothing



function test( a, b = _b)
     for i in 1:100
          b[i] += a + i
     end
     return( b .* 100 )
end

N = 1000
_b = Array{Float64,1}(undef,N)

a = 37.
@btime test(a)


function test!( b, a)
    @inbounds @simd for i in 1:100
        b[i] += a + i
    end
    # Why not reuse "b" again?
    b .*= 100
end

function test_a_lot(N)
    _b = Array{Float64,1}(undef, 100)
    result = 0.0
    for n ∈ 1:N
        result += (-0.01)^n * sum(test!(_b, 0.5n))
    end
    result
end

@btime test_a_lot(5000)
