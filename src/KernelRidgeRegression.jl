
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


# for svgp:
import PyCall: @pyimport, pyimport_conda, PyNULL, PyObject
const np              = PyNULL()
const climin          = PyNULL()
const gpy             = PyNULL()

function __init__()
    copy!(np,     pyimport_conda("numpy", "numpy"))
    copy!(climin, pyimport_conda("climin", "climin"))
    copy!(gpy,    pyimport_conda("GPy", "gpy"))
end
export StochasticVariationalGP
# for svgp end

abstract AbstractKRR{T} <: RegressionModel

function fit(::Type{AbstractKRR}) error("not implemented") end

StatsBase.fitted(KRR::AbstractKRR) = predict(KRR, KRR.X)

include("util.jl")
include("krr.jl")
include("svgp.jl")

end # module KernelRidgeRegression
