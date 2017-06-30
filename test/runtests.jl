
using KernelRidgeRegression
using MLKernels
using Base.Test
using StatsBase

@test GaussianKernel(3.0) == GaussianKernel(3.0)

N = 5000
x = rand(1, N) * 4π - 2π
yy = sinc.(x) # vec(sinc.(4 .* x) .+ 0.2 .* sin.(30 .* x))
y = squeeze(yy + 0.1randn(1, N), 1)
xnew = collect(-2.5π:0.01:2.5π)'

mykrr = fit(KRR, x, y, 1e-3/5000,
            GaussianKernel(100.0))
ynew = predict(mykrr, xnew);

mynystkrr = fit(NystromKRR,
                x, y, 10.0, 280,
                GaussianKernel(100.0))
ynystnew = predict(mynystkrr, xnew)

mysrkrr = fit(SubsetRegressorsKRR,
                x, y, 1.0, 280,
                GaussianKernel(100.0))
ysrnew = predict(mysrkrr, xnew)

myfastkrr = fit(FastKRR,
                x, y, 4/5000, 11,
                GaussianKernel(100.0))
yfastnew = predict(myfastkrr, xnew)

myfastkrr2 = fitPar(FastKRR, length(x),
                    (i) -> x[:, i], (i) -> y[i],
                    4/5000, 11, GaussianKernel(100.0))
yfastnew2 = predict(myfastkrr, xnew)

mytnkrr = fit(TruncatedNewtonKRR,
              x, y, 4/5000, GaussianKernel(100.0),
              0.5, 200)
ytnnew = predict(mytnkrr, xnew)

myrandkrr = fit(RandomFourierFeatures,
                x, y, 1/500.0, 500, 1.0)
yrandnew = predict(myrandkrr, xnew)

emean = sqrt(mean((vec(sinc.(xnew)) - mean(xnew)) .^ 2))
ekrr  = sqrt(mean((vec(sinc.(xnew)) - ynew)       .^ 2))
enyst = sqrt(mean((vec(sinc.(xnew)) - ynystnew)   .^ 2))
esr   = sqrt(mean((vec(sinc.(xnew)) - ysrnew)     .^ 2))
efast = sqrt(mean((vec(sinc.(xnew)) - yfastnew)   .^ 2))
erand = sqrt(mean((vec(sinc.(xnew)) - yrandnew)   .^ 2))
etn   = sqrt(mean((vec(sinc.(xnew)) - ytnnew)     .^ 2))

@test eltype(emean) == Float64
@test eltype(ekrr ) == Float64
@test eltype(enyst) == Float64
@test eltype(esr)   == Float64
@test eltype(efast) == Float64
@test eltype(erand) == Float64
@test eltype(etn  ) == Float64

@test emean > ekrr
@test emean > enyst
@test emean > esr
@test emean > efast
@test emean > erand
@test emean > etn

# from julia/test/show.jl:
replstr(x) = sprint((io, y′) -> show(IOContext(io, :limit => true), MIME("text/plain"), y′), x)

@test replstr(mykrr) == "KernelRidgeRegression.KRR{Float64}:\n    λ = 2.0e-7\n    ϕ = SquaredExponentialKernel(100.0)"
@test contains(replstr(myrandkrr), "KernelRidgeRegression.RandomFourierFeatures{Float64,Complex{Float64}}:\n    λ = 0.002\n:    σ = 1.0\n:    K = 500\n    ϕ = KernelRidgeRegression")
@test replstr(mynystkrr) == "KernelRidgeRegression.NystromKRR{Float64}:\n    λ = 10.0\n    ϕ = SquaredExponentialKernel(100.0)\n    m = 280"
@test replstr(mysrkrr) == "KernelRidgeRegression.SubsetRegressorsKRR{Float64}:\n    λ = 1.0\n    ϕ = SquaredExponentialKernel(100.0)\n    m = 280"
@test replstr(myfastkrr) == "KernelRidgeRegression.FastKRR{Float64}:\n    λ = 0.0008\n    m = 11\n    ϕ = SquaredExponentialKernel(100.0)"
@test replstr(myfastkrr2) == "KernelRidgeRegression.FastKRR{Float64}:\n    λ = 0.0008\n    m = 11\n    ϕ = SquaredExponentialKernel(100.0)"
@test replstr(mytnkrr) == "KernelRidgeRegression.TruncatedNewtonKRR{Float64}:\n    λ = 0.0008\n    ϕ = SquaredExponentialKernel(100.0)"

