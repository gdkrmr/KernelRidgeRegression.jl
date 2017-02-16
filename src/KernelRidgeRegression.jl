module KernelRidgeRegression

export inverse,
    KRR,
    FastKRR,
    RandomFourierFeatures,
    TruncatedNewtonKRR,
    NystromKRR,
    fitPar



# X is dimensions in rows, observations in columns!!!!

import MLKernels
import Iterators
import StatsBase
import StatsBase: fit, fitted, predict, nobs, predict!, RegressionModel

abstract AbstractKRR{T} <: RegressionModel

StatsBase.fitted(KRR::AbstractKRR) = predict(KRR, KRR.X)

include("krr.jl")
include("util.jl")

end # module KernelRidgeRegression
