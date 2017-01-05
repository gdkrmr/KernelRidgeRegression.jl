module KRR

type KRR
    λ :: Real
    K :: Matrix
    κpars :: Tuple = (σ = 0.1, )
end

"""

"""
function fit(::KRR, data::Matrix{Real})
    n, d = size(data)

    
end

"""
Transform distance squared matrix to kernel matrix
"""
function dist_sq_2_gauss_kernel!(Dsq::Matrix, σ::Real)
    n = size(Dsq, 1)
    c = convert(eltype(Dsq), -1/2/σ)
    BLAS.scal!(n^2, c, Dsq, 1)
    @inbounds for i in eachindex(Dsq)
        Dsq[i] = exp(Dsq[i])
    end
    return Dsq
end


"""
Create squared distance matrix

uses ||x - y||^2 = x^2 - 2xy + y^2
"""
function dist_sq(data::Matrix)
    n = size(data, 1)
    Dsq = BLAS.A_mul_Bt(data, data)
    sq = [Dsq[i, i] for i in 1:n]
    cc = ones(sq)
    BLAS.scal!(n^2, -2.0, Dsq, 1)
    BLAS.ger!(1.0, sq, cc, Dsq)
    BLAS.ger!(1.0, sq, cc, Dsq)
    return Dsq
end
end # module KRR

tmp = reshape(collect(1:9.), (3,3))

cross(tmp[1:end], tmp[1:end])


BLAS.ger!(1., collect(1:3.), [1., 1., 1.], tmp)
BLAS.ger!(1., [1., 1., 1.], collect(1:3.), tmp)


(tmp, tmp)
