module KernelRidgeRegression


# X is dimensions in rows, observations in columns!!!!



import MLKernels
import StatsBase
import StatsBase: fit, fitted, predict, nobs

abstract AbstractKRR{T} <: StatsBase.RegressionModel

function StatsBase.fitted(KRR::AbstractKRR)
    predict(KRR, KRR.X)
end

type KRR{T <: AbstractFloat} <: AbstractKRR{T}
    λ :: T
    X :: Matrix{T}
    α :: Vector{T}
    ϕ :: MLKernels.MercerKernel{T}
    function KRR(λ, X, α, ϕ)
        @assert λ > zero(λ)
        @assert size(X, 2) == length(α)
        new(λ, X, α, ϕ)
    end
end

function KRR{T <: AbstractFloat}(
    λ :: T,
    X :: Matrix{T},
    α :: Vector{T},
    ϕ :: MLKernels.MercerKernel{T}
)
    KRR{T}(λ, X, α, ϕ)
end

function fit{T <: AbstractFloat}(
    ::Type{KRR}, X::Matrix{T}, y::Vector{T},
    λ::T, ϕ::MLKernels.Kernel{T}
)
    d, n = size(X)
    K = MLKernels.kernelmatrix!(MLKernels.ColumnMajor(),
                                Matrix{T}(n, n),
                                ϕ, X, true)
    for i = 1:n
        # the n is important to make things comparable between fast and normal
        # KRR
        @inbounds K[i, i] += n * λ
    end

    α = cholfact!(K) \ y

    KRR(λ, X, α, ϕ)
end

function predict{T <: AbstractFloat}(KRR::KRR{T}, X::Matrix{T})
    k = MLKernels.kernelmatrix!(MLKernels.ColumnMajor(),
                                Matrix{T}(size(X, 2), size(KRR.X, 2)),
                                KRR.ϕ, X, KRR.X)
    k * KRR.α
end



type FastKRR{T <: AbstractFloat} <: AbstractKRR{T}
    λ :: T
    n :: Int
    X :: Vector{Matrix{T}}
    α :: Vector{Vector{T}}
    ϕ :: MLKernels.MercerKernel{T}

    function FastKRR(λ, n, X, α, ϕ)
        @assert λ > zero(λ)
        @assert n > zero(n)
        @assert length(X) == length(α)
        for i in 1:length(X)
            @assert size(X[i], 2) == length(α[i])
        end
        new(λ, n, X, α, ϕ)
    end
end

function FastKRR{T <: AbstractFloat}(
    λ :: T,
    n :: Int,
    X :: Vector{Matrix{T}},
    α :: Vector{Vector{T}},
    ϕ :: MLKernels.MercerKernel{T}
)
    FastKRR{T}(λ, n, X, α, ϕ)
end

function StatsBase.fit{T <: AbstractFloat}(
    ::Type{FastKRR}, X::Matrix{T}, y::Vector{T},
    λ::T, nblocks::Int, ϕ::MLKernels.Kernel{T}
)
    d, n = size(X)
    nblocks > n^0.33 && warn("nblocks > n^1/3 = $(n^(1/3)), above theoretical limit")
    nblocks > n^0.45 && warn("nblocks > n^0.45 = $(n^0.45), above empirical limit")

    XX = Vector{Matrix{T}}(nblocks)
    aa = Vector{Vector{T}}(nblocks)

    perm_idxs  = shuffle(1:n)
    blocksizes = make_blocks(n, nblocks)

    b_end   = 0
    for i in 1:nblocks
        b_start = b_end + 1
        b_end  += blocksizes[i]

        i_idxs = perm_idxs[b_start:b_end]
        i_krr  = fit(KRR, X[:, i_idxs], y[i_idxs], λ, ϕ)

        XX[i] = i_krr.X
        aa[i] = i_krr.α
    end
    FastKRR(λ, nblocks, XX, aa, ϕ)
end

fitted(obj::FastKRR) = error("fitted is not defined for $(typeof(obj))")

function predict{T<:AbstractFloat}(FastKRR::FastKRR{T}, X::Matrix{T})
    d, n = size(X)
    pred = zeros(T, n)
    for i in 1:FastKRR.n
        pred += predict(KRR(FastKRR.λ, FastKRR.X[i], FastKRR.α[i], FastKRR.ϕ),  X)
    end
    pred /= FastKRR.n
    pred
end


type RandomKRR{T <: AbstractFloat} <: AbstractKRR{T}
    λ :: T
    K :: Int
    W :: Matrix{T}
    α :: Vector
    Φ :: Function

    function RandomKRR(λ, K, W, α, Φ)
        @assert λ > zero(T)
        @assert K > zero(Int)
        @assert size(W, 2) == K
        new(λ, K, W, α, Φ)
    end
end

function RandomKRR{T}(
    λ :: T,
    K :: Int,
    W :: Matrix{T},
    α :: Vector,
    Φ :: Function
)
    RandomKRR{T}(λ, K, W, α, Φ)
end

function fit{T<:AbstractFloat}(
    ::Type{RandomKRR}, X::Matrix{T}, y::Vector{T},
    λ::T, K::Int, σ::T,
    Φ::Function = (X, W) -> exp(X' * W * 1im) / sqrt(size(W, 2))
)
    d, n = size(X)
    W = randn(d, K)/σ
    Z = Φ(X, W) # Kxd matrix
    # Z = hcat(Z, ones(T, size(Z, 1), 1))
    Z2 = Z' * Z
    for i in 1:K
        @inbounds Z2[i, i] += λ * K
    end
    α = cholfact(Z2) \ (Z' * y)
    RandomKRR(λ, K, W, α, Φ)
end

function predict{T <: AbstractFloat}(RandomKRR::RandomKRR, X::Matrix{T})
    Z = RandomKRR.Φ(X, RandomKRR.W)
    # Z = hcat(Z, ones(T, size(Z, 1), 1))
    real(Z * RandomKRR.α)
end


type TruncatedNewtonKRR{T <: AbstractFloat} <: AbstractKRR{T}
    λ        :: T
    X        :: Matrix{T}
    α        :: Vector{T}
    ϕ        :: MLKernels.MercerKernel{T}
    ɛ        :: T
    max_iter :: Int

    function TruncatedNewtonKRR(λ, X, α, ϕ, ɛ, max_iter)
        @assert size(X, 2) == length(α)
        @assert λ > zero(T)
        @assert ɛ > zero(T)
        @assert max_iter > zero(Int)
        new(λ, X, α, ϕ, ɛ, max_iter)
    end
end

function TruncatedNewtonKRR{T}(
    λ        :: T,
    X        :: Matrix{T},
    α        :: Vector{T},
    ϕ        :: MLKernels.MercerKernel{T},
    ɛ        :: T,
    max_iter :: Int
)
    TruncatedNewtonKRR{T}(λ, X, α, ϕ, ɛ, max_iter)
end

function fit{T <: AbstractFloat}(
    ::Type{TruncatedNewtonKRR}, X::Matrix{T}, y::Vector{T},
    λ::T, ϕ::MLKernels.Kernel{T}, ɛ::T = 0.5, max_iter::Int = 200
)
    d, n = size(X)
    K = MLKernels.kernelmatrix!(MLKernels.ColumnMajor(),
                                Matrix{T}(n, n),
                                ϕ, X, true)
    for i = 1:n
        # the n is important to make things comparable between fast and normal
        # KRR
        @inbounds K[i, i] += n * λ
    end

    α = truncated_newton!(K, y, zeros(y), ɛ, max_iter)

    TruncatedNewtonKRR(λ, X, α, ϕ, ɛ, max_iter)
end

function predict{T<:AbstractFloat}(KRR::TruncatedNewtonKRR{T}, X::Matrix{T})
    k = MLKernels.kernelmatrix!(MLKernels.ColumnMajor(),
                                Matrix{T}(size(X, 2), size(KRR.X, 2)),
                                KRR.ϕ, X, KRR.X)
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
function truncated_newton!{T}(A::Matrix{T}, b::Vector{T},
                              x::Vector{T}, ɛ::T, max_iter::Int)
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
