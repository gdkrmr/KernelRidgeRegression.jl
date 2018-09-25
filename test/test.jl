
using BenchmarkTools
using Gadfly
using StatsBase
BLAS.set_num_threads(1)

using MLKernels

isequal(GaussianKernel(3.0), GaussianKernel(3.0))
GaussianKernel(3.0) == GaussianKernel(3.0)

l = GaussianKernel(3.0)
k = GaussianKernel(3.0)

l == k                         # false
l.alpha == k.alpha             # false
l.alpha.value == k.alpha.value # true

isimmutable(GaussianKernel(3.0)) # true
isimmutable(l.alpha)             # false
isimmutable(l.alpha.bounds)      # true
isimmutable(l.alpha.value)       # true

N = 5000
x = rand(1, N) * 4π - 2π
yy = sinc.(x) # vec(sinc.(4 .* x) .+ 0.2 .* sin.(30 .* x))
y = dropdims(yy .+ 0.1 .* randn(1, N), dims = 1)
xnew = collect(-2.5π:0.01:2.5π)'

reload("KernelRidgeRegression")

@time mykrr = fit(KernelRidgeRegression.KRR, x, y, 1e-3/5000, GaussianKernel(1.0))
@time ynew = predict(mykrr, xnew);

show(mykrr)

@time mynystkrr = fit(KernelRidgeRegression.NystromKRR, x, y, 1e-3/5000, 280, MLKernels.GaussianKernel(100.0))
@time ynystnew = predict(mynystkrr, xnew)

@time myfastkrr = fit(KernelRidgeRegression.FastKRR, x, y, 4/5000, 11, MLKernels.GaussianKernel(100.0))
@time yfastnew = predict(myfastkrr, xnew)

@time mytnkrr = fit(KernelRidgeRegression.TruncatedNewtonKRR, x, y, 4/5000, MLKernels.GaussianKernel(100.0), 0.5, 200)
@time ytnnew = predict(mytnkrr, xnew)

@time myrandkrr = fit(KernelRidgeRegression.RandomFourierFeatures, x, y, 1e-3/5000, 200 , 1.0)
@time yrandnew = predict(myrandkrr, xnew)

# tanh makes the whole thing rotation-symetric and go through the origin
@time myrandkrr2 = fit(KernelRidgeRegression.RandomKRR, x, y,;
                                             4/5000, 2000, 1.0, (X, W) -> tanh(X' * W));
@time yrandnew2 = predict(myrandkrr2, xnew);

@time myrandkrr3 = fit(KernelRidgeRegression.RandomFourierFeatures, x, y,
                                             4/5000, 2000, 1.0, (X, W) -> 1 ./ ((X' * W) .^ 2 + 1))
@time yrandnew3 = predict(myrandkrr3 , xnew);
@time myrandkrr4 = fit(KernelRidgeRegression.RandomFourierFeatures, x, y,
                                             4/5000, 2000, 1.0, (X, W) -> ((X' * W) ./ maximum(X'*W)))
@time yrandnew4 = predict(myrandkrr4 , xnew);
@time myrandkrr5 = fit(KernelRidgeRegression.RandomFourierFeatures, x, y,
                                             4/5000, 2000, 1.0, (X, W) -> begin
                                             z = X' * W
                                             f(x) = x > 0 ? x / (x + 1) : x / (x - 1)
                                             for i in eachindex(z)
                                                 z[i] = f(z[i])
                                             end
                                             z
                                             end)
@time yrandnew5 = predict(myrandkrr5 , xnew);


extrema(ynew - yfastnew)
extrema(ynew - yrandnew)
extrema(ynew - ytnnew)
extrema(ynew - ynystnew)

sqrt(mean((ynew - ynystnew) .^ 2))
sqrt(mean((ynew - yfastnew) .^ 2))
sqrt(mean((ynew - yrandnew) .^ 2))


plot(
    # layer(x = xnew, y = yfastnew,  Geom.line, Theme(default_color = colorant"yellow")),
    # layer(x = xnew, y = ynystnew,  Geom.line, Theme(default_color = colorant"red")),
    # layer(x = xnew, y = ytnnew,    Geom.line, Theme(default_color = colorant"purple")),
    layer(x = xnew, y = yrandnew,  Geom.line, Theme(default_color = colorant"purple")),
    # layer(x = xnew, y = yrandnew2,  Geom.line, Theme(default_color = colorant"yellow")),
    # layer(x = xnew, y = ynew,      Geom.line, Theme(default_color = colorant"green")),
    layer(x = x,    y = y,        Geom.line, Theme(default_color = colorant"blue")),
    Coord.cartesian(ymin = -1.5, ymax = 1.5),
    Guide.manual_color_key(
        "", ["Fast", "Nystrom", "RFF", "Data"],
        [colorant"red", colorant"purple", colorant"green", colorant"blue"]
    )
)


using MLDataUtils
using LossFunctions
using Combinatorics
using DataStructures
reload("KernelRidgeRegression")


N = 500
x = rand(1, N) * 4π - 2π
yy = sinc.(x) # vec(sinc.(4 .* x) .+ 0.2 .* sin.(30 .* x))
y = dropdims(yy .+ 0.1 .* randn(1, N), dims = 1)
xnew = collect(-2.5π:0.01:2.5π)'

reload("KernelRidgeRegression")
KernelRidgeRegression.crossvalidate_parameters(
    KernelRidgeRegression.KRR,
    x, y, 6,
    [10.0^x for x in -5:3],
    [MLKernels.GaussianKernel(10.0^x) for x in -6:3]
)

tmp = Iterators.product([10.0^x for x in -3:3], [10.0^x for x in -3:3])
collect(tmp)
for (k, v) in tmp
    println(k)
    println(v)
end

kk = collect(keys(tmp))
for i in Iterators.product(values(tmp)...)
    @show OrderedDict(kk[j] => i[j] for j = 1:length(i))
end

lossₘᵢₙ
λₘᵢₙ
σₘᵢₙ
ynew = predict(mₘᵢₙ, xnew)



plot(
    # layer(x = xnew, y = ytnnew,    Geom.line, Theme(default_color = colorant"purple")),
    # layer(x = xnew, y = yrandnew,  Geom.line, Theme(default_color = colorant"red")),
    # layer(x = xnew, y = yrandnew2,  Geom.line, Theme(default_color = colorant"yellow")),
    # layer(x = xnew, y = yfastnew,  Geom.line, Theme(default_color = colorant"yellow")),
    layer(x = xnew, y = ynew,      Geom.line, Theme(default_color = colorant"green")),
    # layer(x = x,    y = y,        Geom.line, Theme(default_color = colorant"blue")),
    Coord.cartesian(ymin = -1.5, ymax = 1.5),
    Guide.manual_color_key("", ["RFF", "KRR", "Data"], [colorant"red", colorant"green", colorant"blue"])
)




