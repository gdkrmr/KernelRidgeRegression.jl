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
                x, y, 1e-3/5000, 280,
                GaussianKernel(100.0))
ynystnew = predict(mynystkrr, xnew)

myfastkrr = fit(FastKRR,
                x, y, 4/5000, 11,
                GaussianKernel(100.0))
yfastnew = predict(myfastkrr, xnew)

mytnkrr = fit(TruncatedNewtonKRR,
              x, y, 4/5000, GaussianKernel(100.0),
              0.5, 200)
ytnnew = predict(mytnkrr, xnew)

myrandkrr = fit(RandomFourierFeatures,
                x, y, 1e-3/5000, 200, 1.0)
yrandnew = predict(myrandkrr, xnew)

emean = sqrt(mean((vec(sinc.(xnew)) - mean(xnew))))
ekrr  = sqrt(mean((vec(sinc.(xnew)) - ynew) .^ 2))
enyst = sqrt(mean((vec(sinc.(xnew)) - ynystnew) .^ 2))
efast = sqrt(mean((vec(sinc.(xnew)) - yfastnew) .^ 2))
erand = sqrt(mean((vec(sinc.(xnew)) - yrandnew) .^ 2))
etn   = sqrt(mean((vec(sinc.(xnew)) - ytnnew) .^ 2))

@test emean > ekrr
@test emean > enyst
@test emean > efast
@test emean > erand
@test emean > etn
