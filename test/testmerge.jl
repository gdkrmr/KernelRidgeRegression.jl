# test KRR[...] -> FastKRR
reload("KernelRidgeRegression")

x = randn(10, 1000)
y = randn(1000)

krr_col = map(i -> KernelRidgeRegression.fit(KernelRidgeRegression.KRR, x, y, 1.0, MLKernels.GaussianKernel(1.0)), 1:10)

fast_krr = KernelRidgeRegression.FastKRR(krr_col)

KernelRidgeRegression.predict(fast_krr, x)
