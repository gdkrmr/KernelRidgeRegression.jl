include("src/krr.jl")

using Distances
using BenchmarkTools
using MLKernels
using Regression
using Gadfly

x = rand(1000, 1) * 2 * π - π
y = vec(sinc.(4 .* x) .+ 0.2 .* sin.(30 .* x) .+ 0.1randn(1000))
xnew = collect(-4:0.01:4)''

include("src/krr.jl")
@time mykrr = KRR.krr(x, y, 4., MLKernels.GaussianKernel(100));
@time ynew  = KRR.fit(mykrr, xnew);

plot(layer(x = xnew, y = ynew,    Geom.line, Theme(default_color = colorant"green")),
     layer(x = x,    y = mykrr.α, Geom.line, Theme(default_color = colorant"red")),
     layer(x = x,    y = y,       Geom.line, Theme(default_color = colorant"blue")))




(MLKernels.GaussianKernel <: MLKernels.MercerKernel)
MLKernels.MercerKernel <: MLKernels.Kernel

