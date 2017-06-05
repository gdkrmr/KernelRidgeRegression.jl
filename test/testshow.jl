using KernelRidgeRegression
using StatsBase
using MLKernels

N = 5000
x = rand(1, N) * 4π - 2π
yy = sinc.(x) # vec(sinc.(4 .* x) .+ 0.2 .* sin.(30 .* x))
y = squeeze(yy + 0.1randn(1, N), 1)
xnew = collect(-2.5π:0.01:2.5π)'

@time mykrr = fit(KRR, x, y, 1e-3/5000, GaussianKernel(1.0))
showcompact(mykrr)
show(mykrr)

@time mynystkrr = fit(KernelRidgeRegression.NystromKRR, x, y, 1e-3/5000, 280, MLKernels.GaussianKernel(100.0))
showcompact(mynystkrr)
show(mynystkrr)

@time myfastkrr = fit(KernelRidgeRegression.FastKRR, x, y, 4/5000, 11, MLKernels.GaussianKernel(100.0))
showcompact(myfastkrr)
show(myfastkrr)

@time mytnkrr = fit(KernelRidgeRegression.TruncatedNewtonKRR, x, y, 4/5000, MLKernels.GaussianKernel(100.0), 0.5, 200)
showcompact(mytnkrr)
show(mytnkrr)

@time myrandkrr = fit(KernelRidgeRegression.RandomFourierFeatures, x, y, 1e-3/5000, 200 , 1.0)
showcompact(myrandkrr)
show(myrandkrr)

