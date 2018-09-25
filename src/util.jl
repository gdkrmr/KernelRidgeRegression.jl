# utility functions

function make_blocks(nobs, nblocks)
    maxbs, reminder = divrem(nobs, nblocks)

    res = fill(maxbs, nblocks)
    if reminder > 0
        res[1:reminder] .= maxbs + 1
    end
    res
end

# the truncated newton method for matrix inversion
# adapted from https://en.wikipedia.org/wiki/Conjugate_gradient_method
# solves Ax = b for x, overwrites x
# stops if the error is < ɛ or after reaching max_iter
function truncated_newton!(A::Matrix{T}, b::Vector{T},
                           x::Vector{T}, ɛ::T, max_iter::Int) where T
    r = b - A*x
    p = copy(r)
    Ap = copy(r)
    rsold = dot(r, r)

    n = length(r)

    for i in 1:max_iter
        # Ap[:] = A * p
        mul!(Ap, A, p)
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
