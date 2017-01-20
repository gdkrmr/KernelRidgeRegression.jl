using BenchmarkTools
using Gadfly

x = rand(5000, 1) * 2 * π - π
yy = vec(sinc.(4 .* x) .+ 0.2 .* sin.(30 .* x))
y = yy .+ 0.1randn(5000)
xnew = collect(-4:0.01:4)''

reload("KernelRidgeRegression")

@time mykrr = KernelRidgeRegression.fit(KernelRidgeRegression.KRR, x, y, 4/5000, MLKernels.GaussianKernel(100.0));
@time ynew = KernelRidgeRegression.predict(mykrr, xnew);

@time myfastkrr = KernelRidgeRegression.fit(KernelRidgeRegression.FastKRR, x, y, 4/5000, 10, MLKernels.GaussianKernel(100.0));
@time yfastnew = KernelRidgeRegression.predict(myfastkrr, xnew);

@time mytnkrr = KernelRidgeRegression.fit(KernelRidgeRegression.TruncatedNewtonKRR, x, y, 4/5000, MLKernels.GaussianKernel(100.0), 0.5, 200);
@time ytnnew = KernelRidgeRegression.predict(mytnkrr, xnew);

@time myrandkrr = KernelRidgeRegression.fit(KernelRidgeRegression.RandomKRR, x, y, 4/5000, 500 , 0.01)
@time yrandnew = KernelRidgeRegression.predict(myrandkrr, xnew);

# tanh makes the whole thing rotation-symetric and go through the origin
@time myrandkrr2 = KernelRidgeRegression.fit(KernelRidgeRegression.RandomKRR, x, y,
                                             4/5000, 2000, 1.0, (X, W) -> tanh(X * W))
@time yrandnew2 = KernelRidgeRegression.predict(myrandkrr2, xnew);

@time myrandkrr3 = KernelRidgeRegression.fit(KernelRidgeRegression.RandomKRR, x, y,
                                             4/5000, 2000, 1.0, (X, W) -> 1 ./ ((X * W) .^ 2 + 1))
@time yrandnew3 = KernelRidgeRegression.predict(myrandkrr3 , xnew);
@time myrandkrr4 = KernelRidgeRegression.fit(KernelRidgeRegression.RandomKRR, x, y,
                                             4/5000, 2000, 1.0, (X, W) -> ((X * W) ./ maximum(X*W)))
@time yrandnew4 = KernelRidgeRegression.predict(myrandkrr4 , xnew);
@time myrandkrr5 = KernelRidgeRegression.fit(KernelRidgeRegression.RandomKRR, x, y,
                                             4/5000, 2000, 1.0, (X, W) -> begin
                                             z = X * W
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
    # layer(x = xnew, y = ytnnew,    Geom.line, Theme(default_color = colorant"purple")),
    layer(x = xnew, y = yrandnew5,  Geom.line, Theme(default_color = colorant"red")),
    # layer(x = xnew, y = yfastnew,  Geom.line, Theme(default_color = colorant"yellow")),
    layer(x = xnew, y = ynew,      Geom.line, Theme(default_color = colorant"green")),
    layer(x = x,    y = yy,        Geom.line, Theme(default_color = colorant"blue")),
    Coord.cartesian(ymin = -1.5, ymax = 1.5)
)


include("src/krr.jl")
n = 100
tmp = rand(n, n)
y = rand(n)
tmp1 = ( tmp' + tmp )
for i in 1:n tmp1[i,i] += 10 end
tmp1
tmp2 = deepcopy(tmp1)

@time res1 = cholfact(tmp2) \ y
@time res2 = tmp2 \ y;
@time res3 = KernelRidgeRegression.truncated_newton!(tmp2, y, randn(size(y)), 0.5, 200);
@time res4 = KernelRidgeRegression.truncated_newton2!(tmp2, y, randn(size(y)), 0.5, 200);

res1 - res2
res2 - res3
res2 - res4




(MLKernels.GaussianKernel <: MLKernels.MercerKernel)
MLKernels.MercerKernel <: MLKernels.Kernel

reload("Regression")
Regression.ridgereg(tmp, y, 0)

@code_warntype KernelRidgeRegression.krr(x, y, 4/5000, MLKernels.GaussianKernel(100.0))

module the_foo
type foo{T <: AbstractFloat}
    bar::T
    function foo(bar)
        @assert bar > T(2)
        new(bar)
    end
end
end

