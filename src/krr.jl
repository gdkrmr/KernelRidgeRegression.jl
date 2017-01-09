module KRR

import MLKernels

type krr{T}
    λ :: T
    X :: StridedMatrix{T}
    α :: StridedVector{T}
    ϕ :: MLKernels.MercerKernel{T}
end

function krr{T}(X::StridedMatrix{T}, y::StridedVector{T}, λ::T,
                ϕ::MLKernels.Kernel)
    n, d = size(X)
    K = MLKernels.kernelmatrix(ϕ, X)
    for i = 1:n
        K[i, i] += λ
    end

    α = cholfact!(K) \ y

    krr(λ, X, α, ϕ)
end

function fit{T}(model::krr, X::StridedMatrix{T})
    @show size(X)
    @show size(model.X)
    k = MLKernels.kernelmatrix(model.ϕ, X, model.X)
    @show size(k)
    @show size(model.α)
    k * model.α
end

end # module KRR
