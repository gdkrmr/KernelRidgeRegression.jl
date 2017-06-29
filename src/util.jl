# utility functions

function make_blocks(nobs, nblocks)
    maxbs, reminder = divrem(nobs, nblocks)

    res = fill(maxbs, nblocks)
    if reminder > 0
        res[1:reminder] = maxbs + 1
    end
    res
end

# the truncated newton method for matrix inversion
# adapted from https://en.wikipedia.org/wiki/Conjugate_gradient_method
# solves Ax = b for x, overwrites x
# stops if the error is < ɛ or after reaching max_iter
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
