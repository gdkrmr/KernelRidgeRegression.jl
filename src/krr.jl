module KRR

import MLKernels

type krr{T}
    λ :: T
    X :: StridedMatrix{T}
    α :: StridedVector{T}
    ϕ :: MLKernels.MercerKernel{T}
end


type fast_krr{T}
    λ :: T
    n :: Int
    X :: Array{StridedMatrix, 1}
    α :: Array{StridedVector, 1}
    ϕ :: MLKernels.MercerKernel{T}
end

type random_krr{T}
    λ :: T
    K :: Int
    W :: StridedMatrix{T}
    α :: StridedVector{T}
    Φ :: Function
end

# weighted sum of random kitchen sinks procedure
function random_krr{T}(X::StridedMatrix{T}, y::StridedVector{T}, λ::T,
                       K::Int, σ::T, Φ::Function = (X, W) -> exp(X * W * 1im))
    n, d = size(X)
    W = randn(d, K)/σ
    Z = Φ(X, W) # Kxd matrix
    Z2 = Z' * Z
    for i in 1:K
        @inbounds Z2[i, i] += λ / K
    end
    α = real(cholfact(Z2) \ (Z' * y))
    random_krr(λ, K, W, α, Φ)
end

function fit{T}(random_krr::random_krr, X::StridedMatrix{T})
    real(random_krr.Φ(X, random_krr.W) * random_krr.α)
end

function fast_krr{T}(X::StridedMatrix{T}, y::StridedVector{T}, λ::T,
                     nblocks::Int, ϕ::MLKernels.Kernel{T})
    n, d = size(X)
    if(nblocks > n^0.33) warn("nblocks > n^1/3, above theoretical limit") end
    if(nblocks > n^0.45) warn("nblocks > n^0.45, above empirical limit")  end
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
        i_krr = krr(X[i_idxs, :], y[i_idxs], λ, ϕ)

        XX[i] = i_krr.X
        aa[i] = i_krr.α
    end

    fast_krr(λ, nblocks, XX, aa, ϕ)
end

function fit{T}(fast_krr::fast_krr{T}, X::StridedMatrix{T})
    n, d = size(X)
    pred = zeros(T, n)
    for i in 1:fast_krr.n
        pred += fit(krr(fast_krr.λ, fast_krr.X[i], fast_krr.α[i], fast_krr.ϕ),  X)
    end
    pred /= fast_krr.n
    pred
end


function krr{T}(X::StridedMatrix{T}, y::StridedVector{T}, λ::T,
                ϕ::MLKernels.Kernel{T})
    n, d = size(X)
    K = MLKernels.kernelmatrix(ϕ, X)
    for i = 1:n
        # the n is important to make things comparable between fast and normal
        # krr
        @inbounds K[i, i] += n * λ
    end

    α = cholfact!(K) \ y

    krr(λ, X, α, ϕ)
end

function fit{T}(krr::krr{T}, X::StridedMatrix{T})
    k = MLKernels.kernelmatrix(krr.ϕ, X, krr.X)
    k * krr.α
end

range(x) = minimum(x), maximum(x)

function make_blocks(nobs, nblocks)
    maxbs, rest = divrem(nobs, nblocks)

    res = fill(maxbs, nblocks)
    if rest > 0
        res[1:rest] = maxbs + 1
    end
    res
end

end # module KRR
