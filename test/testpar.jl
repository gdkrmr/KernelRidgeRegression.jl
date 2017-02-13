
using KernelRidgeRegression
using Gadfly
@everywhere using ParallelDataTransfer

N = 5000
x = rand(1, N) * 4π - 2π
yy = sinc.(x) # vec(sinc.(4 .* x) .+ 0.2 .* sin.(30 .* x))
y = squeeze(yy + 0.1randn(1, N), 1)
xnew = collect(-2.5π:0.01:2.5π)'

sendto(workers(), x = x, y = y)

println("KRR")
BLAS.set_num_threads(2)
mykrr = KernelRidgeRegression.fit(KernelRidgeRegression.KRR, x, y, 1e-3/5000, MLKernels.GaussianKernel(1.0))
ynew = KernelRidgeRegression.predict(mykrr, xnew)
@time mykrr = KernelRidgeRegression.fit(KernelRidgeRegression.KRR, x, y, 1e-3/5000, MLKernels.GaussianKernel(1.0))
@time ynew = KernelRidgeRegression.predict(mykrr, xnew)

println("fastKRR")
myfastkrr = KernelRidgeRegression.fit(KernelRidgeRegression.FastKRR, x, y, 4/5000, 10, MLKernels.GaussianKernel(100.0))
yfastnew = KernelRidgeRegression.predict(myfastkrr, xnew)
@time myfastkrr = KernelRidgeRegression.fit(KernelRidgeRegression.FastKRR, x, y, 4/5000, 10, MLKernels.GaussianKernel(100.0))
@time yfastnew = KernelRidgeRegression.predict(myfastkrr, xnew)

println("ParFastKRR")
BLAS.set_num_threads(1)
myparfastkrr = KernelRidgeRegression.fitPar(
    KernelRidgeRegression.FastKRR,
    N,
    i -> x[:,i], i -> y[i],
    4/5000, 10,
    MLKernels.GaussianKernel(100.0)
)
yparfastnew = KernelRidgeRegression.predict(myfastkrr, xnew)
@time myparfastkrr = KernelRidgeRegression.fitPar(
    KernelRidgeRegression.FastKRR, N,
    i -> x[:,i], i -> y[i],
    4/5000, 10,
    MLKernels.GaussianKernel(100.0)
)
@time yparfastnew = KernelRidgeRegression.predict(myfastkrr, xnew)

draw(SVG("test.svg", 20cm, 20cm), plot(
    layer(x = xnew, y = yparfastnew, Geom.line, Theme(default_color = colorant"red")),
    layer(x = xnew, y = yfastnew,    Geom.line, Theme(default_color = colorant"purple")),
    layer(x = xnew, y = ynew,        Geom.line, Theme(default_color = colorant"green")),
    layer(x = x,    y = y,           Geom.line, Theme(default_color = colorant"blue")),
    Coord.cartesian(ymin = -1.5, ymax = 1.5),
    Guide.manual_color_key(
        "Method",
        ["Data",         "KRR",           "FastKRR",        "ParFastKRR" ],
        [colorant"blue", colorant"green", colorant"purple", colorant"red"]
    )
))
