using BenchmarkTools
using Gadfly

N = 5000
x = rand(1, N) * 4π - 2π
yy = sinc.(x) # vec(sinc.(4 .* x) .+ 0.2 .* sin.(30 .* x))
y = squeeze(yy + 0.1randn(1, N), 1)
xnew = collect(-2.5π:0.01:2.5π)'

reload("KernelRidgeRegression")

@time mykrr = KernelRidgeRegression.fit(KernelRidgeRegression.KRR, x, y, 1e-3/5000, MLKernels.GaussianKernel(1.0));
@time ynew = KernelRidgeRegression.predict(mykrr, xnew);

@time mynystkrr = KernelRidgeRegression.fit(KernelRidgeRegression.NystromKRR, x, y, 1e-3/5000, 100, 100, MLKernels.GaussianKernel(100.0));
@time ynystnew = KernelRidgeRegression.predict(mykrr, xnew);

@time myfastkrr = KernelRidgeRegression.fit(KernelRidgeRegression.FastKRR, x, y, 4/5000, 10, MLKernels.GaussianKernel(100.0));
@time yfastnew = KernelRidgeRegression.predict(myfastkrr, xnew);

@time mytnkrr = KernelRidgeRegression.fit(KernelRidgeRegression.TruncatedNewtonKRR, x, y, 4/5000, MLKernels.GaussianKernel(100.0), 0.5, 200);
@time ytnnew = KernelRidgeRegression.predict(mytnkrr, xnew);

@time myrandkrr = KernelRidgeRegression.fit(KernelRidgeRegression.RandomFourierFeatures, x, y, 1e-3/5000, 100 , 1.0);
@time yrandnew = KernelRidgeRegression.predict(myrandkrr, xnew);

# tanh makes the whole thing rotation-symetric and go through the origin
@time myrandkrr2 = KernelRidgeRegression.fit(KernelRidgeRegression.RandomKRR, x, y,;
                                             4/5000, 2000, 1.0, (X, W) -> tanh(X' * W));
@time yrandnew2 = KernelRidgeRegression.predict(myrandkrr2, xnew);

@time myrandkrr3 = KernelRidgeRegression.fit(KernelRidgeRegression.RandomKRR, x, y,
                                             4/5000, 2000, 1.0, (X, W) -> 1 ./ ((X' * W) .^ 2 + 1))
@time yrandnew3 = KernelRidgeRegression.predict(myrandkrr3 , xnew);
@time myrandkrr4 = KernelRidgeRegression.fit(KernelRidgeRegression.RandomKRR, x, y,
                                             4/5000, 2000, 1.0, (X, W) -> ((X' * W) ./ maximum(X'*W)))
@time yrandnew4 = KernelRidgeRegression.predict(myrandkrr4 , xnew);
@time myrandkrr5 = KernelRidgeRegression.fit(KernelRidgeRegression.RandomKRR, x, y,
                                             4/5000, 2000, 1.0, (X, W) -> begin
                                             z = X' * W
                                             f(x) = x > 0 ? x / (x + 1) : x / (x - 1)
                                             for i in eachindex(z)
                                                 z[i] = f(z[i])
                                             end
                                             z
                                             end)
@time yrandnew5 = KernelRidgeRegression.predict(myrandkrr5 , xnew);


KernelRidgeRegression.range(ynew - yfastnew)
KernelRidgeRegression.range(ynew - yrandnew)
KernelRidgeRegression.range(ynew - ytnnew)

plot(
    layer(x = xnew, y = yfastnew,  Geom.line, Theme(default_color = colorant"yellow")),
    # layer(x = xnew, y = ynystnew, Geom.line, Theme(default_color = colorant"red")),
    # layer(x = xnew, y = ytnnew,    Geom.line, Theme(default_color = colorant"purple")),
    # layer(x = xnew, y = yrandnew,  Geom.line, Theme(default_color = colorant"purple")),
    # layer(x = xnew, y = yrandnew2,  Geom.line, Theme(default_color = colorant"yellow")),
    # layer(x = xnew, y = ynew,      Geom.line, Theme(default_color = colorant"green")),
    layer(x = x,    y = y,        Geom.line, Theme(default_color = colorant"blue")),
    Coord.cartesian(ymin = -1.5, ymax = 1.5),
    Guide.manual_color_key("", ["Nystrom", "Fast", "KRR", "Data"], [colorant"red", colorant"purple", colorant"green", colorant"blue"])
)


using MLDataUtils
using LossFunctions
using Combinatorics
using DataStructures
reload("KernelRidgeRegression")


N = 500
x = rand(1, N) * 4π - 2π
yy = sinc.(x) # vec(sinc.(4 .* x) .+ 0.2 .* sin.(30 .* x))
y = squeeze(yy + 0.1randn(1, N), 1)
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




