
# PS6 - Main Script

using Optim, HTTP, GLM, LinearAlgebra, Random
using Statistics, DataFrames, DataFramesMeta, CSV

include("create_grids.jl")
include("PS6_Zhou_source.jl")

function main()

    # Question 1
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS5-ddc/busdata.csv"
    df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)

    println("Question 1: Long panel data")
    println("Observations: ", nrow(df_long))
    println("Variables: ", ncol(df_long))
    println(first(df_long, 5))

    println("Xstate dimensions: ", size(Xstate))
    println("Zstate length: ", length(Zstate))
    println("Branded length: ", length(Branded))

    
# Question 2
println("Question 2: Flexible Logit Model")

flex_logit = estimate_flexible_logit(df_long)

println("Number of coefficients: ", length(coef(flex_logit)))
println("Log likelihood: ", loglikelihood(flex_logit))
println("Model fit completed")


# Question 3(a)
# Construct state transition matrices

zval, zbin, xval, xbin, xtran = create_grids()

println("Question 3(a): State Transition Matrix")
println("zbin: ", zbin)
println("xbin: ", xbin)
println("Transition matrix dimensions: ", size(xtran))
println("All rows sum to 1: ",
        all(isapprox.(sum(xtran, dims=2), 1.0; atol=1e-8)))


# Question 3(b)
# Construct the state space

state_df = construct_state_space(xbin, zbin, xval, zval)

println("Question 3(b): State Space")
println("State space dimensions: ", size(state_df))
println(first(state_df, 5))
println("Last row: ", state_df[end, :])

# Question 3(b): CCP Future Values
FV = compute_future_values(
    state_df, flex_logit, xtran, xbin, 20, 0.9
)

println("Future Value dimensions: ", size(FV))
println("All values finite: ", all(isfinite, FV))
println("Terminal values zero: ", all(FV[:, :, end] .== 0))
println("FV range: ", (minimum(FV), maximum(FV)))


# Question 3(b): Compute FVT1

fvt1 = compute_fvt1(
    df_long, FV, xtran, Xstate, Zstate, xbin, Branded
)

println("FVT1 length: ", length(fvt1))
println("All FVT1 values finite: ", all(isfinite, fvt1))
println("FVT1 range: ", (minimum(fvt1), maximum(fvt1)))
println("First 5 FVT1 values: ", first(fvt1, 5))


# Question 3(c): Estimate Structural Parameters

structural_logit = estimate_structural_params(df_long, fvt1)

println("Question 3(c): Structural Parameter Estimation")
println("Structural parameter estimates:")
println(coeftable(structural_logit))
println("Log likelihood: ", loglikelihood(structural_logit))


    return nothing
end

@time main()