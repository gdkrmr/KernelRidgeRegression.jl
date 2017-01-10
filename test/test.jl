include("src/krr.jl")

using Distances
using BenchmarkTools
using MLKernels
using Regression
using Gadfly

x = rand(5000, 1) * 2 * π - π
yy = vec(sinc.(4 .* x) .+ 0.2 .* sin.(30 .* x))
y = yy .+ 0.1randn(5000)
xnew = collect(-4:0.01:4)''

include("src/krr.jl")
@time mykrr     = KRR.krr(       x, y, 4/5000,      MLKernels.GaussianKernel(100.0));
@time ynew     = KRR.fit(mykrr, xnew)

@time myfastkrr = KRR.fast_krr(  x, y, 4/5000, 10, MLKernels.GaussianKernel(100.0));
@time yfastnew = KRR.fit(myfastkrr, xnew)

@time myrandkrr = KRR.random_krr(x, y, 4/5000, 5000 , 0.01)
@time yrandnew = KRR.fit(myrandkrr, xnew)

KRR.range(ynew - yfastnew)
KRR.range(ynew - yrandnew)

plot(
    layer(x = xnew, y = yrandnew, Geom.line, Theme(default_color = colorant"red")),
    layer(x = xnew, y = yfastnew, Geom.line, Theme(default_color = colorant"yellow")),
    layer(x = xnew, y = ynew,     Geom.line, Theme(default_color = colorant"green")),
    layer(x = x,    y = yy,       Geom.line, Theme(default_color = colorant"blue")),
    Coord.cartesian(ymin = -2, ymax = 2)
)


tmp = rand(10, 10)
y = rand(10)
tmp1 = ( tmp' + tmp )
for i in 1:10 tmp1[i,i] += 10 end
tmp2 = deepcopy(tmp1)
@time cholfact!(tmp1)
@time tmp3 = cholfact(tmp2)
tmp1
tmp3

cholfact!(tmp1) \ y
cholfact(tmp2) \ y


(MLKernels.GaussianKernel <: MLKernels.MercerKernel)
MLKernels.MercerKernel <: MLKernels.Kernel

reload("Regression")
Regression.ridgereg(tmp, y, 0)

