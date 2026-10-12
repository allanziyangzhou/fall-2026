
# PS5 - Question 1
# Reshape data from wide to long panel format

using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM
using LinearAlgebra
using Optim

include("create_grids.jl")

include("PS5_Zhou_source.jl")

function main()

# Question 1
df_long = load_static_data()

# Display the dimensions and first five rows
println("Question 1: Long-format panel data")
println("Number of observations: ", nrow(df_long))
println("Number of variables: ", ncol(df_long))
println(first(df_long, 5))

# Question 2
model = estimate_static_model(df_long)

println("Question 2: Static binary logit model")
show(stdout, MIME"text/plain"(), model)
println()

# Question 3(a)
d = load_dynamic_data()

println("Question 3(a): Dynamic model data")
println("Y dimensions: ", size(d.Y))
println("X dimensions: ", size(d.X))
println("Xstate dimensions: ", size(d.Xstate))
println("Zstate length: ", length(d.Zstate))

# Question 3(b)
zval, zbin, xval, xbin, xtran = create_grids()

println("Question 3(b): Transition matrix")
println("zbin: ", zbin)
println("xbin: ", xbin)
println("Transition matrix dimensions: ", size(xtran))

# Question 3(c)
FV = zeros(d.zbin * d.xbin, 2, d.T + 1)

θ = coef(model)
compute_future_value!(FV, θ, d)

println("Question 3(c): Future value function")
println("FV dimensions: ", size(FV))
println("Terminal values are zero: ", all(FV[:, :, end] .== 0))

# Question 3(d)
loglike = compute_log_likelihood(θ, d, FV)

println("Question 3(d): Log likelihood")
println("Log likelihood: ", loglike)

# Question 3(h)
println("Question 3(h): Dynamic model estimation")

result = optimize(
    θ -> negative_log_likelihood(θ, d),
    coef(model),
    LBFGS(),
    Optim.Options(
        iterations=100,
        g_tol=1e-5,
        show_trace=true,
        show_every=1
    )
)

println("Estimated parameters: ", Optim.minimizer(result))
println("Log likelihood: ", -Optim.minimum(result))
println("Converged: ", Optim.converged(result))
println("Iterations: ", Optim.iterations(result))
println("Optimization summary: ", result)

    return nothing
end

main()