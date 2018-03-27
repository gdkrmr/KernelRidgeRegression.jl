


"""
Gaussian Processes for big data.

currently uses GPy
"""
type StochasticVariationalGP <: AbstractKRR
    o :: PyObject
    p :: PyObject
end


function fit(::Type{StochasticVariationalGP}, X, y, m, batchsize, n_steps)

    d, n = size(X)
    @assert length(y) == n

    m = gpy[:core][:SVGP](
        X',                         # GPy uses observations in columns
        reshape(y, (n, 1)),         # GPy can have more than one output variable
        # memory for the support vectors initialized with asubset of X
        X[: , sample(1:n, m)]',
        gpy[:kern][:RBF](d) + gpy[:kern][:White](d),
        gpy[:likelihoods][:Gaussian]();
        batchsize = batchsize
    )

    opt = climin[:Adadelta](
        m[:optimizer_array],
        m[:stochastic_grad],
        step_rate = 0.5,
        momentum = 0.9
    )

    # klc = Float64[]
    # kva = Float64[]
    # gva = Float64[]
    # err = Float64[]

    i = 0
    for o in opt
        # n_batch = m[:new_batch]()
        # y_batch = n_batch[2]
        # x_batch = n_batch[1]
        # y_batch_hat = m[:predict](x_batch)[1]
        # rmse = (y_batch - y_batch_hat) .^ 2 |> mean |> sqrt
        # push!(kva, m[:kern][:variance][1])
        # push!(klc, m[:kern][:lengthscale][1])
        # push!(gva, m[:Gaussian_noise][:variance][1])
        # push!(err, rmse)
        # println("$i: err = $(err[end]); lc = $(lc[end]); va = $(va[end])")
        i += 1

        if i % 100 == 0
            @show i
        end

        if i > n_steps
            break
        end
    end

    opt = climin[:Adadelta](
        m[:optimizer_array],
        m[:stochastic_grad],
        step_rate = 0.05,
        momentum  = 0.9
    )

    i = 0
    for o in opt
        i += 1
        if i > n_steps
            break
        end
    end

    opt = climin[:Adadelta](
        m[:optimizer_array],
        m[:stochastic_grad],
        step_rate = 0.01,
        momentum  = 0.9
    )

    i = 0
    for o in opt
        i += 1
        if i > n_steps
            break
        end
    end

    StochasticVariationalGP(opt, m)
end

function fit!(m::StochasticVariationalGP, n_steps)
    klc = Float64[]
    kva = Float64[]
    gva = Float64[]
    err = Float64[]

    i = 0
    for o in m.opt
        n_batch = m[:new_batch]()
        y_batch = n_batch[2]
        x_batch = n_batch[1]
        y_batch_hat = m[:predict](x_batch)[1]

        rmse = (y_batch - y_batch_hat) .^ 2 |> mean |> sqrt
        push!(kva, m[:kern][:variance][1])
        push!(klc, m[:kern][:lengthscale][1])
        push!(gva, m[:Gaussian_noise][:variance][1])
        push!(err, rmse)
        # println("$i: err = $(err[end]); lc = $(lc[end]); va = $(va[end])")

        i += 1
        if i > n_steps
            break
        end
    end
    nothing
end

function predict(m::StochasticVariationalGP, X)
    m.p[:predict](X')[1] |> vec
end

