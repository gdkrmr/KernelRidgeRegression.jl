include("src/krr.jl")

using Distances
using BenchmarkTools
using MLKernels
using Gadfly

x = rand(5000, 1) * 2 * π - π
yy = vec(sinc.(4 .* x) .+ 0.2 .* sin.(30 .* x))
y = yy .+ 0.1randn(5000)
xnew = collect(-4:0.01:4)''

include("src/krr.jl")
@time mykrr     = KRR.krr(       x, y, 4/5000,      MLKernels.GaussianKernel(100.0));
@time ynew     = KRR.fit(mykrr, xnew);

@time myfastkrr = KRR.fast_krr(  x, y, 4/5000, 10, MLKernels.GaussianKernel(100.0));
@time yfastnew = KRR.fit(myfastkrr, xnew);

@time mytnkrr = KRR.tn_krr(x, y, 4/5000, MLKernels.GaussianKernel(100.0); ɛ = 0.5, max_iter = 200);
@time ytnnew = KRR.fit(mytnkrr, xnew);

@time myrandkrr = KRR.random_krr(x, y, 4/5000, 500 , 0.01);
@time yrandnew = KRR.fit(myrandkrr, xnew);

# tanh makes the whole thing anti-symetric and go through the origin
@time myrandkrr2 = KRR.random_krr(x, y, 4/5000, 2000, 1.0, (X, W) -> tanh(X * W))
@time yrandnew2 = KRR.fit(myrandkrr2, xnew);

@time myrandkrr3 = KRR.random_krr(x, y, 4/5000, 2000, 1.0, (X, W) -> 1 ./ ((X * W) .^ 2 + 1))
@time yrandnew3 = KRR.fit(myrandkrr3, xnew);

KRR.range(ynew - yfastnew)
KRR.range(ynew - yrandnew)
KRR.range(ynew - ytnnew)

plot(
    layer(x = xnew, y = ytnnew,    Geom.line, Theme(default_color = colorant"purple")),
    # layer(x = xnew, y = yrandnew,  Geom.line, Theme(default_color = colorant"red")),
    # layer(x = xnew, y = yfastnew,  Geom.line, Theme(default_color = colorant"yellow")),
    layer(x = xnew, y = ynew,      Geom.line, Theme(default_color = colorant"green")),
    # layer(x = x,    y = yy,        Geom.line, Theme(default_color = colorant"blue")),
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
@time res3 = KRR.truncated_newton!(tmp2, y, randn(size(y)), 0.5, 200);
@time res4 = KRR.truncated_newton2!(tmp2, y, randn(size(y)), 0.5, 200);

res1 - res2
res2 - res3
res2 - res4




(MLKernels.GaussianKernel <: MLKernels.MercerKernel)
MLKernels.MercerKernel <: MLKernels.Kernel

reload("Regression")
Regression.ridgereg(tmp, y, 0)

@code_warntype KRR.krr(x, y, 4/5000, MLKernels.GaussianKernel(100.0))
