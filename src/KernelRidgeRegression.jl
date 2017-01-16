module KernelRidgeRegression

import MLKernels
import StatsBase
import StatsBase: fit, fitted, predict

abstract AbstractKRR <: StatsBase.RegressionModel

function fitted(KRR::AbstractKRR)
    predict(KRR, KRR.X)
end

type KRR{T <: AbstractFloat} <: AbstractKRR
    λ :: T
    X :: StridedMatrix{T}
    α :: StridedVector{T}
    ϕ :: MLKernels.MercerKernel{T}
end

function KRR{T}(
    λ::T, X::Array{StridedMatrix, 1},
    α::Array{StridedVector, 1}, ϕ::MLKernels.MercerKernel{T}
)
    @assert λ > zero(T)
    @assert size(X, 1) == length(α)
    FastKRR(λ, X, α, ϕ)
end


type FastKRR{T <: AbstractFloat} <: AbstractKRR
    λ :: T
    n :: Int
    X :: Array{StridedMatrix, 1}
    α :: Array{StridedVector, 1}
    ϕ :: MLKernels.MercerKernel{T}
end

function FastKRR{T <: AbstractFloat}(
    λ::T, n::Int, X::Array{StridedMatrix, 1},
    α::Array{StridedVector, 1}, ϕ::MLKernels.MercerKernel{T}
)
    @assert λ > zero(T)
    @assert n > zero(Int)
    @assert length(X) == length(α)
    for i in 1:length(X)
        @assert size(X[i], 1) == length(α[i])
    end
    FastKRR(λ, n, X, α, ϕ)
end


type RandomKRR{T <: AbstractFloat} <: AbstractKRR
    λ :: T
    K :: Int
    W :: StridedMatrix{T}
    α :: StridedVector{T}
    Φ :: Function
end

function RandomKRR{T<:AbstractFloat}(
    λ::T, K::Int, W::StridedMatrix{T},
    α::StridedVector{T}, Φ::Functon
)
    @assert λ > zero(T)
    @assert K > zero(Int)
    @assert size(W, 1) == length(α)
    RandomKRR(λ, K, W, α, Φ)
end


type TruncNewtKRR{T <: AbstractFloat}
    λ :: T
    X :: StridedMatrix{T}
    α :: StridedVector{T}
    ϕ :: MLKernels.MercerKernel{T}
    ɛ :: T
    max_iter :: Int
end

function TruncNewtKRR{T<:AbstractFloat}(
    λ::T, X::StridedMatrix{T}, α::StridedVector{T},
    ϕ::MLKernels.MercerKernel{T}, ɛ::T, max_iter::Int
)
    @assert size(X, 1) == length(α)
    @assert λ > zero(T)
    @assert ɛ > zero(T)
    @assert max_iter > zero(Int)
    TruncNewtKRR(λ, X, α, ϕ, ɛ, max_iter)
end

function fit{T<:AbstractFloat}(
    Type{TruncNewtKRR}, X::StridedMatrix{T}, y::StridedVector{T},
    λ::T, ϕ::MLKernels.Kernel{T}, ɛ::T = 0.5, max_iter::Int = 200
)
    n, d = size(X)
    K = MLKernels.kernelmatrix(ϕ, X)
    for i = 1:n
        # the n is important to make things comparable between fast and normal
        # KRR
        @inbounds K[i, i] += n * λ
    end

    α = truncated_newton!(K, y, zeros(y), ɛ, max_iter)

    TruncNewtKRR(λ, X, α, ϕ, ɛ, max_iter)
end

function predict{T<:AbstractFloat}(KRR::TruncNewtKRR{T}, X::StridedMatrix{T})
    k = MLKernels.kernelmatrix(KRR.ϕ, X, KRR.X)
    k * KRR.α
end



function fit{T<:AbstractFloat}(
    Type{RandomKRR{T}}, X::StridedMatrix{T}, y::StridedVector{T},
    λ::T, K::Int, σ::T, Φ::Function = (X, W) -> exp(X * W * 1im)
)
    n, d = size(X)
    W = randn(d, K)/σ
    Z = Φ(X, W) # Kxd matrix
    Z2 = Z' * Z
    for i in 1:K
        @inbounds Z2[i, i] += λ / K
    end
    α = real(cholfact(Z2) \ (Z' * y))
    RandomKRR(λ, K, W, α, Φ)
end

function predict{T <: AbstractFloat}(RandomKRR::RandomKRR, X::StridedMatrix{T})
    real(RandomKRR.Φ(X, RandomKRR.W) * RandomKRR.α)
end



function fit{T <: AbstractFloat}(
    Type{FastKRR{T}}, X::StridedMatrix{T}, y::StridedVector{T},
    λ::T, nblocks::Int, ϕ::MLKernels.Kernel{T}
)
    n, d = size(X)
    nblocks > n^0.33 && warn("nblocks > n^1/3, above theoretical limit ($(n^(1/3)))")
    nblocks > n^0.45 && warn("nblocks > n^0.45, above empirical limit ($(n^0.45))")
    XX = Vector{StridedMatrix}(nblocks)
    aa = Vector{StridedVector}(nblocks)

    perm_idxs  = shuffle(1:n)
    blocksizes = make_blocks(n, nblocks)

    b_start = 1
    b_end   = 0
    for i in 1:nblocks
        b_start = b_end + 1
        b_end  += blocksizes[i]
        i_idxs = perm_idxs[b_start:b_end]
        i_krr = KRR(X[i_idxs, :], y[i_idxs], λ, ϕ)

        XX[i] = i_krr.X
        aa[i] = i_krr.α
    end

    FastKRR(λ, nblocks, XX, aa, ϕ)
end

fitted(obj::FastKRR) = error("fitted is not defined for $(typeof(obj))")

function predict{T<:AbstractFloat}(FastKRR::FastKRR{T}, X::StridedMatrix{T})
    n, d = size(X)
    pred = zeros(T, n)
    for i in 1:FastKRR.n
        pred += fit(KRR(FastKRR.λ, FastKRR.X[i], FastKRR.α[i], FastKRR.ϕ),  X)
    end
    pred /= FastKRR.n
    pred
end


function fit{T <: AbstractFloat}(
    Type{KRR{T}}, X::StridedMatrix{T}, y::StridedVector{T}, λ::T,
    ϕ::MLKernels.Kernel{T}
)
    n, d = size(X)
    K = MLKernels.kernelmatrix(ϕ, X)
    for i = 1:n
        # the n is important to make things comparable between fast and normal
        # KRR
        @inbounds K[i, i] += n * λ
    end

    α = cholfact!(K) \ y

    KRR(λ, X, α, ϕ)
end

function predict{T <: AbstractFloat}(KRR::KRR{T}, X::StridedMatrix{T})
    k = MLKernels.kernelmatrix(KRR.ϕ, X, KRR.X)
    k * KRR.α
end



# utility functions 
range(x) = minimum(x), maximum(x)

function make_blocks(nobs, nblocks)
    maxbs, rest = divrem(nobs, nblocks)

    res = fill(maxbs, nblocks)
    if rest > 0
        res[1:rest] = maxbs + 1
    end
    res
end

# the truncated newton method for matrix inversion
# adapted from https://en.wikipedia.org/wiki/Conjugate_gradient_method
# solves Ax = b for x, overwrites x
function truncated_newton!{T}(A::StridedMatrix{T}, b::StridedVector{T},
                              x::StridedVector{T}, ɛ::T, max_iter::Int)
    r = b - A*x
    p = deepcopy(r)
    Ap = deepcopy(r)
    rsold = dot(r, r)

    n = length(r)

    for i in 1:max_iter
        # Ap[:] = A * p
        A_mul_B!(Ap, A, p)
        α = rsold / dot(p, Ap)
        # x += α * p
        BLAS.axpy!(α, p, x)
        # r -= α * Ap
        BLAS.axpy!(-α, Ap, r)
        rsnew = dot(r, r)
        rsnew < ɛ &&  break
        β = rsnew / rsold
        # p[:] = r + β * p
        BLAS.scal!(n, β, p, 1)
        BLAS.axpy!(1, r, p)
        rsold = rsnew
    end
    return x
end

end # module KernelRidgeRegression
