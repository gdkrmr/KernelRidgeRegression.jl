
# X is dimensions in rows, observations in columns!!!!

"""
Basic Kernel Ridge Regression.

* `λ`: The regularization parameter.
* `X`: The data, a matrix with dimensions in rows and observations in columns.
* `α`: The weights of the linear regression in kernel space, will be calculated by `fit`.
* `ϕ`: A Kernel function
"""
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

function predict!{T <: AbstractFloat}(
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

function predict!{T <: AbstractFloat}(KRR::KRR{T}, X::Matrix{T}, y::Vector{T})
    predict!(
        KRR, X, y, MLKernels.kernelmatrix!(
            MLKernels.ColumnMajor(),
            Matrix{T}(size(X, 2), size(KRR.X, 2)),
            KRR.ϕ, X, KRR.X
    ))
end

function predict{T <: AbstractFloat}(KRR::KRR{T}, X::Matrix{T})
    predict!(KRR, X, Vector{T}(size(X, 2)))
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

function showcompact(io::IO, x::KRR)
    show(io, typeof(x))
end

function show(io::IO, x::KRR)
    showcompact(io, x)
    print(io, ":\n    λ = ", x.λ)
    print(io,  "\n    ϕ = "); show(io, x.ϕ)
end


"""
Fast Kernel Ridge Regression.

Divides the problem in `m` splits and calculates a separate Kernel Ridge Regression for each.

* `λ`: The regularization parameter.
* `m`: The number of splits for the data.
* `X`: A vector containing a data matrix for each split.
* `α`: A vector containing the weights of the linear regressions in kernel space for each split,
       will be calculated by `fit`.
* `ϕ`: A Kernel function (not a vector of kernel functions!).
"""
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
        (nₘₐₓ - nₘᵢₙ) > 1 && warn(
            "number of observations per block should not differ by more than one"
        )
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
            ((krrs[i].ϕ != ϕ) ||
             (krrs[i].λ != λ)) &&
             error("all kernel functions and λs must be the same")
        end
    end

    FastKRR{T}(λ, m, X, α, ϕ)
end

# equality hack for MLKernels
# not fixed in 0.1.0
# import Base.==
# ==(x::MLKernels.Kernel, y::MLKernels.Kernel) =
#     error("not implemented for types $(typeof(x)), $(typeof(y))")
# ==(x::MLKernels.HyperParameters.HyperParameter, y::MLKernels.HyperParameters.HyperParameter) =
#     x.value == y.value
# ==(x::MLKernels.GaussianKernel, y::MLKernels.GaussianKernel) =
#     x.alpha == y.alpha

function fit{T <: AbstractFloat}(
      :: Type{FastKRR},
    X :: Matrix{T},
    y :: Vector{T},
    λ :: T,
    m :: Int,
    ϕ :: MLKernels.Kernel{T}
)
    d, n = size(X)
    # Those are the limits for polynomial kernels,
    # the gaussian kernel needs a little bit less blocks
    m > n^0.33 && warn("m > n^1/3 = $(n^(1/3)), above theoretical limit")
    m > n^0.45 && warn("m > n^0.45 = $(n^0.45), above empirical limit")

    XX = Vector{Matrix{T}}(m)
    aa = Vector{Vector{T}}(m)

    perm_idxs  = shuffle(1:n)
    blocksizes = make_blocks(n, m)

    b_end = 0
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
Fit a FastKRR in parallel

* `n`:     The total number of observations.
* `get_X`: A function that given a vector of observation indices will load these into memory,
            e.g.:
            `get_X(inds) = X[:, inds]`
            `get_y(inds) = y[inds]`
* `get_y`: A a function which given a vector of response indices will load these into memory.
* `λ`:     The regularization parameter.
* `m`:     The number of splits for the data.
* `ϕ`:     A Kernel function
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
    # Those are the limits for polynomial kernels,
    # the gaussian kernel needs a little bit less blocks
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

fitted(obj::FastKRR) = error("fitted is not defined for $(typeof(obj))")

function predict{T <: AbstractFloat}(fast_krr::FastKRR{T}, X::Matrix{T})
    @assert fast_krr.m > 0
    d, n = size(X)
    y = zeros(T, n)
    K = Matrix{T}(n, size(fast_krr.X[1], 2))
    for i in 1:fast_krr.m

        # The KRR.X[i] may be of different lengths
        if size(K, 2) != size(fast_krr.X[i], 2)
            K = Matrix{T}(n, size(fast_krr.X[i], 2))
        end

        predict_and_add!(
            KRR(fast_krr.λ, fast_krr.X[i], fast_krr.α[i], fast_krr.ϕ),
            X, y, K
        )
    end

    for i in 1:n
        @inbounds y[i] = y[i] / fast_krr.m
    end

    return y
end

function showcompact(io::IO, x::FastKRR)
    show(io, typeof(x))
end

function show(io::IO, x::FastKRR)
    showcompact(io, x)
    println(io, ":\n    λ = ", x.λ)
    println(io,    "    m = ", x.m)
    print(io,      "    ϕ = "); show(io, x.ϕ)
end


"""
Random Fourier Features

Details see Rahimi and Recht (2008)

* `λ`: The regularization parameter.
* `K`: The number of random vectors.
* `W`: The random weights.
* `α`: A vector containing the weights of the linear regressions in kernel space for each split,
       will be calculated by `fit`.
* `ϕ`: Kernel approximation function function.
"""
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
    σ :: T
    W :: Matrix{T},
    α :: Vector{S},
    ϕ :: Function
)
    RandomFourierFeatures{T, S}(λ, K, W, α, ϕ)
end

function fit{T<:AbstractFloat}(
      :: Type{RandomFourierFeatures},
    X :: Matrix{T},
    y :: Vector{T},
    λ :: T,
    K :: Int,
    σ :: T,
    ϕ :: Function = (X, W) -> exp(X' * W * 1im)
)
    d, n = size(X)
    W = randn(d, K) / σ
    Z = ϕ(X, W) / sqrt(K) # Kxd matrix, the normalization can probably be dropped
    Z2 = Z' * Z
    for i in 1:K
        @inbounds Z2[i, i] += λ * K
    end
    α = cholfact(Z2) \ (Z' * y)
    RandomFourierFeatures(λ, K, σ, W, α, ϕ)
end

function predict{T <: AbstractFloat}(RFF::RandomFourierFeatures, X::Matrix{T})
    Z = RFF.ϕ(X, RFF.W) / sqrt(RFF.K)
    real(Z * RFF.α)
end

function showcompact(io::IO, x::RandomFourierFeatures)
    show(io, typeof(x))
end

function show(io::IO, x::RandomFourierFeatures)
    showcompact(io, x)
    println(io, ":\n    λ = ", x.λ)
    println(io,   ":    σ = ", x.σ)
    println(io,   ":    K = ", x.K)
    print(io,      "    ϕ = "); show(io, x.ϕ)
end

"""
Truncated Newton Kernel Ridge Regression

Approximates the Kernel Ridge Regression by an early stopped optimization

* `λ`: The regularization parameter.
* `X`: The data, a matrix with dimensions in rows and observations in columns.
* `α`: The weights of the linear regression in kernel space, will be calculated by `fit`.
* `ϕ`: A Kernel function
* `ɛ`: Error stopping criterion
* `max_iter`: Maximum number of iterations.
"""
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

function showcompact(io::IO, x::TruncatedNewtonKRR)
    show(io, typeof(x))
end

function show(io::IO, x::TruncatedNewtonKRR)
    showcompact(io, x)
    print(io, ":\n    λ = ", x.λ)
    print(io,  "\n    ϕ = "); show(io, x.ϕ)
end

"""
Nystrom Approximation of a Kernel Ridge Regression

* `λ`:  The regularization parameter.
* `Xm`: The sampled data, a matrix with dimensions in rows and observations in columns.
* `m`:  The number of samples.
* `ϕ`: A Kernel function
* `α`:  The weights of the linear regression in kernel space, will be calculated by `fit`.
"""
type NystromKRR{T <: AbstractFloat} <: AbstractKRR{T}
    λ  :: T
    Xm :: Matrix{T}
    m  :: Integer
    ϕ  :: MLKernels.MercerKernel{T}
    α  :: Vector{T}

    function NystromKRR(λ, Xm, m, ϕ, α)
        @assert m == size(Xm, 2)
        @assert λ >= zero(λ)
        @assert length(α) == m
        new(λ, Xm, m, ϕ, α)
    end
end

function NystromKRR{T <: AbstractFloat}(
    λ  :: T,
    Xm :: Matrix{T},
    m  :: Integer,
    ϕ  :: MLKernels.MercerKernel{T},
    α  :: Vector{T}
)
    NystromKRR{T}(λ, Xm, m, ϕ, α)
end

function fit{T <: AbstractFloat}(
      :: Type{NystromKRR},
    X :: Matrix{T},
    y :: Vector{T},
    λ :: T,
    m :: Integer,
    ϕ :: MLKernels.MercerKernel{T}
)
    d, n = size(X)
    @assert m < n
    m_idx = sample(1:n, m, replace = false)
    Xm = X[:, m_idx]
    Kmn = MLKernels.kernelmatrix!(MLKernels.ColumnMajor(),
                                  Matrix{T}(m, n),
                                  ϕ, Xm, X)
    Kmm = Kmn * Kmn'
    for i in 1:m
        @inbounds Kmm[i, i] += m * λ
    end

    α = cholfact(Kmm) \ (Kmn * y)
    NystromKRR(λ, Xm, m, ϕ, α)
end

function predict{T <: AbstractFloat}(KRR :: NystromKRR{T}, X :: Matrix{T})
    Knm = MLKernels.kernelmatrix!(MLKernels.ColumnMajor(),
                                  Matrix{T}(size(X, 2), size(KRR.Xm, 2)),
                                  KRR.ϕ, X, KRR.Xm)
    Knm * KRR.α
end

function showcompact(io::IO, x::NystromKRR)
    show(io, typeof(x))
end

function show(io::IO, x::NystromKRR)
    showcompact(io, x)
    print(io, ":\n    λ = ", x.λ)
    print(io,  "\n    ϕ = "); show(io, x.ϕ)
    print(io,  "\n    m = ", x.m)
end

# An implementation error which nonetheless works
type SomethingKRR{T <: AbstractFloat} <: AbstractKRR{T}
    λ    :: T
    X    :: Matrix{T}  # The data d × n
    r    :: Integer    # the rank, <= m
    m    :: Integer    # the number of samples
    ϕ    :: MLKernels.MercerKernel{T}
    α    :: Vector{T}  # Weight vector n × 1
    Σinv :: Vector{T}  # Standard deviations length r
    Vt   ::  Matrix{T} # Eigenvectors

    function SomethingKRR(λ, X, r, m, ϕ, α, Σinv, Vt)
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

function SomethingKRR{T}(
    λ    :: T,
    X    :: Matrix{T},
    r    :: Integer,
    m    :: Integer,
    ϕ    :: MLKernels.MercerKernel{T},
    α    :: Vector{T},
    Σinv :: Vector{T},
    Vt   :: Matrix{T}
)
    SomethingKRR{T}(λ, X, r, m, ϕ, α, Σinv, Vt)
end

function fit{T <: AbstractFloat}(
      :: Type{SomethingKRR},
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

    sᵢ = sample(1:n, m, replace = false)
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

    return SomethingKRR(λ, X, r, m, ϕ, α, Σinv, Vt)
end

function predict{T <: AbstractFloat}(KRR :: SomethingKRR{T}, Xnew :: Matrix{T})
    d, n = size(Xnew)
    Kbnew = MLKernels.kernelmatrix!(MLKernels.ColumnMajor(),
                                    Matrix{T}(size(KRR.X, 2), n),
                                    KRR.ϕ, KRR.X, Xnew)
    KRR.alpha' * KRR.Σinv * KRR.Vt * Kbnew
end
