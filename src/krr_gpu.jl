# gpu versn = 20000
x = randn(10, n)
using GPUArrays
using CUBLAS

function gauss{T}(X::Matrix{T}, alpha::T)
    n = size(X, 2)
    xy = LinAlg.BLAS.syrk('U', 'T', one(T), X)
    x2 = [ sum(X[:, i] .^ 2) for i in 1:n ]

    LinAlg.BLAS.syr2k!('U', 'N', one(T), x2, ones(T, n), T(-2), xy)

    @inbounds for i in 1:n
        for j in 1:i
            xy[j, i] = exp(-alpha * xy[j, i])
        end
    end
    LinAlg.copytri!(xy, 'U')
end
gauss(x, 1.0)ions

type KRR_GPU{T <: AbstractFloat} <: AbstractKRR{T}
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

function StatsBase.fit{T <: AbstractFloat}(
      :: Type{KRR},
    X :: Matrix{T},
    y :: Vector{T},
    λ :: T,
    ϕ :: MLKernels.Kernel{T}
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



function StatsBase.predict!{T <: AbstractFloat}(
    KRR :: KRR{T},
    X   :: Matrix{T},
    y   :: Vector{T},
    K   :: Matrix{T}
)
    n, n_new = (size(KRR.X, 2), size(X, 2))
    @assert (n_new, n) == size(K)
    @assert length(y) == n_new

    MLKernels.kernelmatrix!(
        MLKernels.ColumnMajor(),
        K, KRR.ϕ, X, KRR.X
    )
    LinAlg.BLAS.gemv!('N', one(T), K, KRR.α, zero(T), y)
    return y
end

function StatsBase.predict!{T <: AbstractFloat}(KRR::KRR{T}, X::Matrix{T}, y::Vector{T})
    StatsBase.predict!(
        KRR, X, y, MLKernels.kernelmatrix!(
            MLKernels.ColumnMajor(),
            Matrix{T}(size(X, 2), size(KRR.X, 2)),
            KRR.ϕ, X, KRR.X
    ))
end

function StatsBase.predict{T <: AbstractFloat}(KRR::KRR{T}, X::Matrix{T})
    StatsBase.predict!(KRR, X, Vector{T}(size(X, 2)))
end

# predict the KRR with the data in X and add the result to y, used to speedup
# predict(fast_krr::FastKRR)
function predict_and_add!{T <: AbstractFloat}(
    KRR :: KRR{T},
    X   :: Matrix{T},
    y   :: Vector{T},
    K   :: Matrix{T}
)
    n, n_new = (size(KRR.X, 2), size(X, 2))
    @assert (n_new, n) == size(K)
    @assert length(y) == n_new

    MLKernels.kernelmatrix!(
        MLKernels.ColumnMajor(),
        K, KRR.ϕ, X, KRR.X
    )
    LinAlg.BLAS.gemv!('N', one(T), K, KRR.α, one(T), y)
    return y
end
