module KernelRidgeRegression

export inverse,
    KRR,
    FastKRR,
    RandomFourierFeatures,
    TruncatedNewtonKRR,
    NystromKRR



# X is dimensions in rows, observations in columns!!!!

import MLKernels
import Iterators
import StatsBase
import StatsBase: fit, fitted, predict, nobs, predict!, RegressionModel

abstract AbstractKRR{T} <: RegressionModel

function fitted(KRR::AbstractKRR)
    predict(KRR, KRR.X)
end

include("krr.jl")
include("util.jl")

end # module KernelRidgeRegression
