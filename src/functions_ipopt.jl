# obj(nlp, x): evaluate the objective value at x;
# grad!(nlp, x, g): evaluate the objective gradient at x;
# cons!(nlp, x, c): evaluate the vector of constraints, if any;
# jac_structure!(nlp, rows, cols): fill rows and cols with the sparsity structure of the Jacobian, if the problem is constrained;
# jac_coord!(nlp, x, vals): fill vals with the Jacobian values corresponding to the sparsity structure returned by jac_structure!();
# hess_structure!(nlp, rows, cols): fill rows and cols with the sparsity structure of the lower triangle of the Hessian of the Lagrangian;
# hess_coord!(nlp, x, hvals; obj_weight=1) # unconstrained
# hess_coord!(nlp, x, y, vals; obj_weight=1.0): constrained needs y -- fill vals with the values of the Hessian of the Lagrangian corresponding to the sparsity structure returned by hess_structure!(), where obj_weight is the weight assigned to the objective, and y is the vector of multipliers.

# cons_lin!(nlp, x, c)  # c = cons_lin(nlp, x)
# jac_lin_structure!(nlp, jrows, jcols)
# jac_lin_coord!(nlp, x, jvals)
# hess_coord!(nlp, x, y, hvals; obj_weight=1)

mutable struct modcon{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    A::Matrix{T}
    b::Vector{T}
end

function modcon(
    A::Matrix{T}, b::Vector{T};
    nvar::Int=size(A)[1], ncon::Int=length(b), 
    x0::Vector{T}=ones(nvar), 
    lvar::Vector{T}=fill(-Inf, nvar), uvar::Vector{T}=fill(Inf, nvar),
    lcon::Vector{T}=fill(-Inf, ncon), ucon::Vector{T}=fill(Inf, ncon), # lin::Vector{Int}=collect(1:ncon),
    nnzh::Int=nvar, nnzj::Int=count(!iszero, A), lin_nnzj=nnzj) where T

    # Create the NLPModelMeta instance
    meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, lcon=lcon, ucon=ucon, lin=collect(1:ncon), 
           nnzh=nnzh, nnzj=nnzj, lin_nnzj=nnzj) # , nln_nnzj=0

    # Create the Counters instance
    counters = Counters()

    # Create and return the modMyModel instance
    return modcon(meta, counters, A, b)
end


# Objective function
function NLPModels.obj(nlp::modcon, x::AbstractVector)
    return sum((1 .- x).^2)
end
  
  # Gradient of the objective function
function NLPModels.grad!(nlp::modcon, x::AbstractVector, g::AbstractVector)
    g .= -2 .* (1 .- x)
    return g
end

# Hessian structure
function NLPModels.hess_structure!(nlp::modcon, hrows::AbstractVector{<:Integer}, hcols::AbstractVector{<:Integer})
    hrows .= 1:nlp.meta.nvar
    hcols .= 1:nlp.meta.nvar
    return hrows, hcols
  end
  
# Hessian values
function NLPModels.hess_coord!(nlp::modcon, x::AbstractVector, y::AbstractVector, hvals::AbstractVector; obj_weight::Real=1.0)
    # As the hessian is constant and doesn't depend on x or y, we ignore these parameters
    hvals .= 2.0 * obj_weight
    return hvals
end

# constraints  
function NLPModels.cons_lin!(nlp::modcon, x::AbstractVector, c::AbstractVector)
    c .= nlp.A' * x # - nlp.b
    return c
end  

# Jacobian structure
function NLPModels.jac_structure!(nlp::modcon, jrows::AbstractVector{<:Integer}, jcols::AbstractVector{<:Integer})
    # If the constraints are linear, the Jacobian structure is the same as the A matrix non-zero structure
    # r, c = findnz(sparse(nlp.A)) - old BAD
    r, c = findnz(sparse(nlp.A'))
    jrows .= r
    jcols .= c
    return jrows, jcols
end

# Jacobian values
function NLPModels.jac_coord!(nlp::modcon, x::AbstractVector, jvals::AbstractVector)
    # If the constraints are linear, the Jacobian values are the same as the A matrix non-zero values
    jvals .= nonzeros(sparse(nlp.A))
    return jvals
end
