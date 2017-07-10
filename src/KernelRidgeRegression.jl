
__precompile__()
module KernelRidgeRegression

export KRR,
    FastKRR,
    RandomFourierFeatures,
    TruncatedNewtonKRR,
    NystromKRR,
    SubsetRegressorsKRR,
    fitPar

import Base: show, showcompact, display
import MLKernels
import StatsBase
import StatsBase: fit, fitted, predict, nobs, predict!, RegressionModel, sample, weights

abstract AbstractKRR{T} <: RegressionModel

function fit(::Type{AbstractKRR}) error("not implemented") end

StatsBase.fitted(KRR::AbstractKRR) = predict(KRR, KRR.X)

include("krr.jl")
include("util.jl")

end # module KernelRidgeRegression
