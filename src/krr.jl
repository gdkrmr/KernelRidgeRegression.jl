
# X is dimensions in rows, observations in columns!!!!

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

function StatsBase.predict{T <: AbstractFloat}(KRR::KRR{T}, X::Matrix{T})
    k = MLKernels.kernelmatrix!(MLKernels.ColumnMajor(),
                                Matrix{T}(size(X, 2), size(KRR.X, 2)),
                                KRR.ϕ, X, KRR.X)
    k * KRR.α
end



type FastKRR{T <: AbstractFloat} <: AbstractKRR{T}
    λ :: T
    m :: Int
    X :: Vector{Matrix{T}}
    α :: Vector{Vector{T}}
    ϕ :: MLKernels.MercerKernel{T}

    function FastKRR(λ, m, X, α, ϕ)
        @assert λ > zero(λ)
        @assert m > zero(m)
        @assert length(X) == length(α)
        nₘᵢₙ = Inf
        nₘₐₓ = 0
        for i in 1:length(X)
            n = size(X[i], 2)
            @assert n == length(α[i])
            (nₘᵢₙ > n) && (nₘᵢₙ = n)
            (nₘₐₓ < n) && (nₘₐₓ = n)
        end
        (nₘₐₓ - nₘᵢₙ) > 1 && warn("number of observations per block should not differ by more than one")
        new(λ, m, X, α, ϕ)
    end
end

function FastKRR{T <: AbstractFloat}(
    λ :: T,
    m :: Int,
    X :: Vector{Matrix{T}},
    α :: Vector{Vector{T}},
    ϕ :: MLKernels.MercerKernel{T}
)
    FastKRR{T}(λ, m, X, α, ϕ)
end

function FastKRR{T <: AbstractFloat}(krrs :: Union{Vector{KRR{T}}, Tuple{KRR{T}}})
    m = length(krrs)

    λ = krrs[1].λ
    X = map((i) -> krrs[i].X, 1:m)
    α = map((i) -> krrs[i].α, 1:m)
    ϕ = krrs[1].ϕ

    if m > 1
        for i in 2:m
            ((krrs[i].ϕ == ϕ) ||
             (krrs[i].λ == λ)) &&
             error("all kernel functions and λs must be the same")
        end
    end

    FastKRR{T}(λ, m, X, α, ϕ)
end

# equality hack for MLKernels
import Base.==
==(x::MLKernels.Kernel, y::MLKernels.Kernel) = error("not implemented")
==(x::MLKernels.HyperParameters.HyperParameter, y::MLKernels.HyperParameters.HyperParameter) = x.value == y.value
==(x::MLKernels.GaussianKernel, y::MLKernels.GaussianKernel) = x.alpha == y.alpha

function StatsBase.fit{T <: AbstractFloat}(
      :: Type{FastKRR},
    X :: Matrix{T},
    y :: Vector{T},
    λ :: T,
    m :: Int,
    ϕ :: MLKernels.Kernel{T}
)
    d, n = size(X)
    # Those are the limits for polynomial kernels, the gaussian kernel needs a little bit less blocks
    m > n^0.33 && warn("m > n^1/3 = $(n^(1/3)), above theoretical limit")
    m > n^0.45 && warn("m > n^0.45 = $(n^0.45), above empirical limit")

    XX = Vector{Matrix{T}}(m)
    aa = Vector{Vector{T}}(m)

    perm_idxs  = shuffle(1:n)
    blocksizes = make_blocks(n, m)

    b_end   = 0
    for i in 1:m
        b_start = b_end + 1
        b_end  += blocksizes[i]

        i_idxs = perm_idxs[b_start:b_end]
        i_krr  = fit(KRR, X[:, i_idxs], y[i_idxs], λ, ϕ)

        XX[i] = i_krr.X
        aa[i] = i_krr.α
    end
    FastKRR(λ, m, XX, aa, ϕ)
end

"""
get_X is a function that given a list of indices will load these into memory,
e.g.:
get_X(inds) = X[:,inds]
get_y(inds) = y[inds]
"""
function fitPar{T <: AbstractFloat}(
          :: Type{FastKRR},
    n     :: Int,
    get_X :: Function,
    get_y :: Function,
    λ     :: T,
    m     :: Int,
    ϕ     :: MLKernels.Kernel{T}
)
    # Those are the limits for polynomial kernels, the gaussian kernel needs a little bit less blocks
    m > n^0.33 && warn("m > n^1/3 = $(n^(1/3)), above theoretical limit")
    m > n^0.45 && warn("m > n^0.45 = $(n^0.45), above empirical limit")

    perm_idxs  = shuffle(1:n)
    blocksizes = make_blocks(n, m)
    b_starts = [1; cumsum(blocksizes)[1:end-1]+1]
    b_ends   = cumsum(blocksizes)

    krrs = pmap((i) -> fit(
        KRR,
        get_X(perm_idxs[b_starts[i]:b_ends[i]]),
        get_y(perm_idxs[b_starts[i]:b_ends[i]]),
        λ, ϕ
    ), 1:m)

    XX = map((i) -> krrs[i].X, 1:m)
    aa = map((i) -> krrs[i].α, 1:m)

    FastKRR(λ, m, XX, aa, ϕ)
end

StatsBase.fitted(obj::FastKRR) = error("fitted is not defined for $(typeof(obj))")

function StatsBase.predict{T<:AbstractFloat}(FastKRR::FastKRR{T}, X::Matrix{T})
    d, n = size(X)
    pred = zeros(T, n)
    # predᵢ = zeros(T, n)
    for i in 1:FastKRR.m
        pred += StatsBase.predict(KRR(FastKRR.λ, FastKRR.X[i], FastKRR.α[i], FastKRR.ϕ),  X)
        # TODO: need a predict! function !!
        # predict!(KRR(FastKRR.λ, FastKRR.X[i], FastKRR.α[i], FastKRR.ϕ), predᵢ,  X)
        # BLAS.axpy!(1.0, predᵢ, pred)
    end
    pred /= FastKRR.m
    pred
end


type RandomFourierFeatures{T <: AbstractFloat, S <: Number} <: AbstractKRR{T}
    λ :: T
    K :: Int
    W :: Matrix{T}
    α :: Vector{S}
    ϕ :: Function

    function RandomFourierFeatures(λ, K, W, α, ϕ)
        @assert λ >= zero(T)
        @assert K > zero(Int)
        @assert size(W, 2) == K
        new(λ, K, W, α, ϕ)
    end
end

function RandomFourierFeatures{T <: AbstractFloat, S <: Number}(
    λ :: T,
    K :: Int,
    W :: Matrix{T},
    α :: Vector{S},
    ϕ :: Function
)
    RandomFourierFeatures{T, S}(λ, K, W, α, ϕ)
end

function StatsBase.fit{T<:AbstractFloat}(
      :: Type{RandomFourierFeatures},
    X :: Matrix{T},
    y :: Vector{T},
    λ :: T,
    K :: Int,
    σ :: T,
    ϕ :: Function = (X, W) -> exp(X' * W * 1im)
)
    d, n = size(X)
    W = randn(d, K)/σ
    Z = ϕ(X, W) / sqrt(K) # Kxd matrix, the normalization can probably be dropped
    Z2 = Z' * Z
    for i in 1:K
        @inbounds Z2[i, i] += λ * K
    end
    α = cholfact(Z2) \ (Z' * y)
    RandomFourierFeatures(λ, K, W, α, ϕ)
end

function StatsBase.predict{T <: AbstractFloat}(RFF::RandomFourierFeatures, X::Matrix{T})
    Z = RFF.ϕ(X, RFF.W) / sqrt(RFF.K)
    real(Z * RFF.α)
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

function StatsBase.fit{T <: AbstractFloat}(
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

function StatsBase.predict{T<:AbstractFloat}(KRR::TruncatedNewtonKRR{T}, X::Matrix{T})
    k = MLKernels.kernelmatrix!(MLKernels.ColumnMajor(),
                                Matrix{T}(size(X, 2), size(KRR.X, 2)),
                                KRR.ϕ, X, KRR.X)
    k * KRR.α
end


type NystromKRR{T <: AbstractFloat} <: AbstractKRR{T}
    λ :: T
    X :: Matrix{T}    # The data d × n
    r :: Integer      # the rank, <= m
    m :: Integer      # the number of samples
    ϕ :: MLKernels.MercerKernel{T}
    α :: Vector{T}    # Weight vector n × 1
    Σinv :: Vector{T} # Standard deviations length r
    Vt ::  Matrix{T}  # Eigenvectors

    function NystromKRR(λ, X, r, m, ϕ, α, Σinv, Vt)
        d, n = size(X)
        @assert 0 <  r
        @assert r <= m
        @assert m <= n
        @assert λ >= zero(λ)
        @assert r == length(α)
        @assert r == length(Σinv)
        @assert r == size(Vt, 2)
        @assert m == size(Vt, 1)
        new(λ, X, r, m, ϕ, α, Σinv, Vt)
    end
end

function NystromKRR{T}(
    λ    :: T,
    X    :: Matrix{T},
    r    :: Integer,
    m    :: Integer,
    ϕ    :: MLKernels.MercerKernel{T},
    α    :: Vector{T},
    Σinv :: Vector{T},
    Vt   :: Matrix{T}
)
    NystromKRR{T}(λ, X, r, m, ϕ, α, Σinv, Vt)
end

function StatsBase.fit{T <: AbstractFloat}(
      :: Type{NystromKRR},
    X :: Matrix{T},
    y :: Vector{T},
    λ :: T,
    m :: Integer,
    r :: Integer,
    ϕ :: MLKernels.Kernel{T}
)
    d, n = size(X)
    @assert 0 < r
    @assert r <= m
    @assert m <= n
    @assert λ >= zero(λ)

    sᵢ = StatsBase.sample(1:n, m, replace = false)
    Xₛ = X[:, sᵢ]
    Kb = MLKernels.kernelmatrix!(MLKernels.ColumnMajor(),
                                 Matrix{T}(m, n),
                                 ϕ, Xₛ, X)
    K = Kb[:, sᵢ]
    # @assert issymmetric(K)
    USVt = svdfact(K)

    ord = sortperm(USVt.S, rev = true)[1:r]
    Σinv = 1 ./ USVt.S[ord]
    Vt = USVt.Vt[ord, :]

    α = Diagonal( (λ*r) .+ Σinv ) * Vt * Kb * y

    return NystromKRR(λ, X, r, m, ϕ, α, Σinv, Vt)
end

function StatsBase.predict{T <: AbstractFloat}(KRR :: NystromKRR{T}, Xnew :: Matrix{T})
    d, n = size(Xnew)
    Kbnew = MLKernels.kernelmatrix!(MLKernels.ColumnMajor(),
                                    Matrix{T}(size(KRR.X, 2), n),
                                    KRR.ϕ, KRR.X, Xnew)
    KRR.alpha' * KRR.Σinv * KRR.Vt * Kbnew
end
