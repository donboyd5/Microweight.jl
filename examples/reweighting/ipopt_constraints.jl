
# add constraints --------------------------------------------------------------------------------------------------
using LinearAlgebra
using NLPModels, NLPModelsIpopt, SparseArrays

hsllib = "/usr/local/lib/lib/x86_64-linux-gnu/libhsl.so"

include("mtprw.jl")

mutable struct mmcon{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    A::Matrix{T}
    b::Vector{T}
end

function mmcon(nvar::Int, 
    A::Matrix{T}, b::Vector{T};
    x0::Vector{T}=ones(nvar), lvar::Vector{T}=fill(-Inf, nvar), uvar::Vector{T}=fill(Inf, nvar),
    ncon::Int=length(b), lcon::Vector{T}=fill(-Inf, ncon), ucon::Vector{T}=fill(Inf, ncon), # lin::Vector{Int}=collect(1:ncon),
    nnzh::Int=nvar, nnzj::Int=count(!iszero, A), lin_nnzj=nnzj) where T

    # Create the NLPModelMeta instance
    # println(lvar)
    meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, lcon=lcon, ucon=ucon, lin=collect(1:ncon), 
           nnzh=nnzh, nnzj=nnzj, lin_nnzj=nnzj, nln_nnzj=0)

    # Create the Counters instance
    counters = Counters()

    # Create and return the modMyModel instance
    return mmcon(meta, counters, A, b)
end

# obj(nlp, x)
# grad!(nlp, x, g)
# hess_structure!(nlp, hrows, hcols)
# hess_coord!(nlp, x, hvals; obj_weight=1) # unconstrained
# hess_coord!(nlp, x, y, hvals; obj_weight=1) # constrained -- needs y

# cons_lin!(nlp, x, c)  # c = cons_lin(nlp, x)
# jac_lin_structure!(nlp, jrows, jcols)
# jac_lin_coord!(nlp, x, jvals)
# hess_coord!(nlp, x, y, hvals; obj_weight=1)

# obj(nlp, x): evaluate the objective value at x;
# grad!(nlp, x, g): evaluate the objective gradient at x;
# cons!(nlp, x, c): evaluate the vector of constraints, if any;
# jac_structure!(nlp, rows, cols): fill rows and cols with the spartity structure of the Jacobian, if the problem is constrained;
# jac_coord!(nlp, x, vals): fill vals with the Jacobian values corresponding to the sparsity structure returned by jac_structure!();
# hess_structure!(nlp, rows, cols): fill rows and cols with the spartity structure of the lower triangle of the Hessian of the Lagrangian;
# hess_coord!(nlp, x, y, vals; obj_weight=1.0): fill vals with the values of the Hessian of the Lagrangian corresponding to the sparsity structure returned by hess_structure!(), where obj_weight is the weight assigned to the objective, and y is the vector of multipliers.



# Objective function
function NLPModels.obj(nlp::mmcon, x::AbstractVector)
    return sum((1 .- x).^2)
end
  
  # Gradient of the objective function
function NLPModels.grad!(nlp::mmcon, x::AbstractVector, g::AbstractVector)
    g .= -2 .* (1 .- x)
    return g
end

# Hessian structure
function NLPModels.hess_structure!(nlp::mmcon, hrows::AbstractVector{<:Integer}, hcols::AbstractVector{<:Integer})
    hrows .= 1:nlp.meta.nvar
    hcols .= 1:nlp.meta.nvar
    return hrows, hcols
  end
  
# Hessian values
function NLPModels.hess_coord!(nlp::mmcon, x::AbstractVector, y::AbstractVector, hvals::AbstractVector; obj_weight::Real=1.0)
    # As the hessian is constant and doesn't depend on x or y, we ignore these parameters
    hvals .= 2.0 * obj_weight
    return hvals
end

# constraints  
function NLPModels.cons_lin!(nlp::mmcon, x::AbstractVector, c::AbstractVector)
    c .= nlp.A' * x - nlp.b
    return c
end  

# Jacobian structure
function NLPModels.jac_structure!(nlp::mmcon, jrows::AbstractVector{<:Integer}, jcols::AbstractVector{<:Integer})
    # If the constraints are linear, the Jacobian structure is the same as the A matrix non-zero structure
    r, c = findnz(sparse(nlp.A))
    jrows .= r
    jcols .= c
    return jrows, jcols
end

# Jacobian values
function NLPModels.jac_coord!(nlp::mmcon, x::AbstractVector, jvals::AbstractVector)
    # If the constraints are linear, the Jacobian values are the same as the A matrix non-zero values
    jvals .= nonzeros(sparse(nlp.A))
    return jvals
end


# n = 5
# A = Matrix{Float64}(I, n, 3)
# A = [1. 2. 0.; 
#      4. 0. 6.;
#      0. 8. 9.;
#      10. 11. 12.;
#      13. 14. 15.]


# A = Matrix{Float64}(I, n, n)
# A = [1. 2. 0.; 
#      4. 0. 6.;
#      0. 8. 9.]
A = [1. 2.; 
     4. 0.;
     0. 8.]

n = size(A)[1]
findnz(sparse(A))
x = ones(n)
lvar = fill(0.1, n)
uvar = fill(10.0, n)

A' * x
b = fill(2.0, size(A)[2]) # ones(2)
lcon=b.*0.9
ucon=b.*1.1

# b = [4., 11.]
# #b = [27., 36., 41.]
# b = [4., 11., 14.]
b

mod2 = mmcon(n, A, b)

# res = ipopt(mod2, print_level=5, hessian_constant="yes")

mod2 = mmcon(n, A, b; lvar=lvar, uvar=uvar, lcon=lcon, ucon=ucon)

# constraints at x0
mod2.A' * mod2.meta.x0 - mod2.b
# jac
mod2.A
r, c = findnz(sparse(mod2.A))
r
c
# jac values
nonzeros(sparse(mod2.A))



mod2 = mmcon(n, A, b; lcon=b.*0.9, ucon=b.*1.1)
mod2.meta.nvar
mod2.meta.ncon
mod2.meta.lvar
mod2.meta.uvar
mod2.meta.lcon
mod2.meta.ucon
mod2.meta.nnzh
mod2.meta.nnzj
mod2.meta.lin
mod2.A
mod2.b

mod2.A' * mod2.meta.x0 - mod2.b

rows = zeros(Int, mod2.meta.nnzh)
cols = zeros(Int, mod2.meta.nnzh)
vals = zeros(mod2.meta.nnzh)
NLPModels.hess_structure!(mod2, rows, cols)
NLPModels.hess_coord!(mod2, rows, cols, vals)

rows = zeros(Int, mod2.meta.nnzj)
cols = zeros(Int, mod2.meta.nnzj)
vals = zeros(mod2.meta.nnzj)
NLPModels.jac_structure!(mod2, rows, cols)
NLPModels.jac_coord!(mod2, mod2.meta.x0, vals)
mod2.A

cx = zeros(mod2.meta.ncon)
NLPModels.cons_lin!(mod2, mod2.meta.x0, cx)

res = ipopt(mod2, print_level=5, hessian_constant="yes")

# mod2 = mmcon(n, A, b, lvar=fill(0.1, n), uvar=fill(10.0, n), ncon=n, lcon=b, ucon=b)
mod2 = mmcon(n, A, b, lcon=b, ucon=b)


mod2 = mmcon(n, A, b, lvar=lvar, uvar=uvar, ncon=n, lcon=b, ucon=b)


res = ipopt(mod2, print_level=5, hessian_constant="yes")
res.objective
res.solution

res = ipopt(mod2, print_level=5, hessian_constant="yes", hsllib=hsllib, linear_solver="ma57")

# try with simulated data
tp = mtprw(100, 4, pctzero=.3)
tp = mtprw(10, 2, pctzero=0)
fieldnames(typeof(tp))
tp.h
tp.rwtargets
tp.rwtargets_calc

n = tp.h
A = tp.xmat .* tp.wh
b = tp.rwtargets
nnzj = count(!iszero, A)
lb = fill(0.1, n)
ub = fill(10.0, n)
x = ones(tp.h)
A' * x - b
tp.rwtargets
tp.rwtargets_calc


mod3 = mmcon(n, A, b, lvar=lb, uvar=ub, lcon=b, ucon=b, nnzj=nnzj)
# fieldnames(typeof(mod3.meta))
mod3.meta.nvar
mod3.meta.ncon
mod3.meta
mod3.meta.lin

res = ipopt(mod3, print_level=5, hessian_constant="yes")
# fieldnames(typeof(res))
res.status
res.objective
res.solution


nlp = mod2
nlp.A * x - nlp.b


mod = mod3

println("Size of A: ", size(mod.A))
println("Length of b: ", length(mod.b))
println("Number of variables: ", mod.meta.nvar)
println("Number of constraints: ", mod.meta.ncon)
println("Size of x0: ", length(mod.meta.x0))
println("Size of lvar: ", length(mod.meta.lvar))
println("Size of uvar: ", length(mod.meta.uvar))

x = rand(mod.meta.nvar)
g = zeros(mod.meta.nvar)
NLPModels.grad!(mod, x, g)
println("Size of gradient: ", length(g))

rows = zeros(Int, mod.meta.nnzh)
cols = zeros(Int, mod.meta.nnzh)
vals = zeros(mod.meta.nnzh)
NLPModels.hess_structure!(mod, rows, cols)
NLPModels.hess_coord!(mod, x, vals)
println("Size of Hessian structure: ", length(rows), ", ", length(cols))
println("Size of Hessian values: ", length(vals))


