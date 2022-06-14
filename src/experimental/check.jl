

f2(x) = x.^4 .- x.^2
f2([1., 2., 3.])



# GOOD:
x0 = [1., 2., 3.]
r = f2(x0)
r2 = OnceDifferentiable(f2, x0, copy(r); inplace = false, autodiff = :forward)
lsres = LsqFit.levenberg_marquardt(r2, x0, show_trace = true)
lsres = LsqFit.lmfit(f2, [1., 2., 3.], Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=50)


# Good
f100(x, y, z) = x .* y  .+ z
f200 = x -> f100(x, y, z)
f200([3., 4., 5.])
y = 2.
z = 3.
x100 = [1., 2., 3.]
r100 = f200(x100)
r200 = OnceDifferentiable(f200, x100, copy(r100); inplace = false, autodiff = :forward)
lsres = LsqFit.levenberg_marquardt(r200, x100, show_trace = true)
fieldnames(typeof(lsres))

lsres.method
lsres.minimizer
lsres.minimum
f100(lsres.minimizer, y, z)

using Parameters

@with_kw mutable struct temp
    # https://discourse.julialang.org/t/default-value-of-some-fields-in-a-mutable-struct/33408
    minimizer::Vector{Float64} = [Inf]
    minimum::Float64 = Inf
end

a = temp()
a.minimizer = [1.0, 2.0, 3.0]
a.minimum = 0.0
a


@with_kw mutable struct Result
    # https://discourse.julialang.org/t/default-value-of-some-fields-in-a-mutable-struct/33408
    method::Symbol = :missing
    minimizer::Vector{Float64} = [Inf]
    minimum::Float64 = Inf
    solver_result = nothing
end

a = Result(method=:lm)

b = Result()
a.method = :lmfit