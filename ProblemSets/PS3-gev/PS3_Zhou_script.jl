# ============================================================
# PS3_Zhou_script.jl
# ECON 6343 - Problem Set 3
# ============================================================


# ============================================================
# Load packages
# ============================================================

using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using CSV
using FreqTables


# ============================================================
# Load function definitions
# ============================================================

include(joinpath(@__DIR__, "PS3_Zhou_source.jl"))


# ============================================================
# Main function
# ============================================================

function run_ps3()

    # ========================================================
    # Load data
    # ========================================================

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS3-gev/nlsw88w.csv"

    df = CSV.read(HTTP.get(url).body, DataFrame)


    # ========================================================
    # Construct variables
    # ========================================================

    X = hcat(
        df.age,
        df.white,
        df.collgrad
    )

    Z = hcat(
        df.elnwage1,
        df.elnwage2,
        df.elnwage3,
        df.elnwage4,
        df.elnwage5,
        df.elnwage6,
        df.elnwage7,
        df.elnwage8
    )

    y = df.occupation

    x_names = [
        "age",
        "white",
        "collgrad"
    ]


    # ========================================================
    # Question 1: Multinomial Logit
    # ========================================================

    result_q1,
    theta_hat_q1,
    beta_hat_q1,
    gamma_hat_q1 =
        estimate_mnl(X, Z, y)


    occupation_names = [
        "Professional/Technical",
        "Managers/Administrators",
        "Sales",
        "Clerical/Unskilled",
        "Craftsmen",
        "Operatives",
        "Transport"
    ]


    println()
    println("================================================")
    println("Question 1: Multinomial Logit")
    println("================================================")


    for j in 1:7

        println()
        println("Alternative $j: $(occupation_names[j])")

        for k in 1:3
            println("  $(x_names[k]) = $(beta_hat_q1[k, j])")
        end

    end


    println()
    println("Alternative 8: Other")
    println("  beta = 0 (normalized)")


    println()
    println("Estimated gamma:")
    println("  gamma = $gamma_hat_q1")


    println()
    println("Optimization:")
    println("  Converged = $(Optim.converged(result_q1))")
    println("  Negative log-likelihood = $(Optim.minimum(result_q1))")


    # ========================================================
    # Question 2: Interpretation of gamma
    # ========================================================

    # The estimated coefficient is gamma = -0.0942.
    #
    # Holding the individual-specific characteristics X_i fixed,
    # a one-unit increase in Z_ij - Z_iJ decreases the log-odds
    # of choosing occupation j relative to the normalized
    # occupation J by approximately 0.0942.
    #
    # Therefore, a larger value of Z for occupation j relative
    # to occupation J is associated with a lower probability
    # of choosing occupation j relative to occupation J,
    # holding the other covariates fixed.


    # ========================================================
    # Question 3: Nested Logit
    # ========================================================

    result_q3,
    theta_hat_q3,
    beta_WC_hat_q3,
    beta_BC_hat_q3,
    lambda_WC_hat_q3,
    lambda_BC_hat_q3,
    gamma_hat_q3 =
        estimate_nested_logit(X, Z, y)


    println()
    println("================================================")
    println("Question 3: Nested Logit")
    println("================================================")


    println()
    println("White Collar Nest (WC) beta coefficients:")

    for k in 1:3
        println("  $(x_names[k]) = $(beta_WC_hat_q3[k])")
    end


    println()
    println("Blue Collar Nest (BC) beta coefficients:")

    for k in 1:3
        println("  $(x_names[k]) = $(beta_BC_hat_q3[k])")
    end


    println()
    println("Inclusive value parameters:")
    println("  lambda_WC = $lambda_WC_hat_q3")
    println("  lambda_BC = $lambda_BC_hat_q3")


    println()
    println("Estimated gamma:")
    println("  gamma = $gamma_hat_q3")


    println()
    println("Other occupation:")
    println("  beta_Other = 0 (normalized)")


    println()
    println("Optimization:")
    println("  Converged = $(Optim.converged(result_q3))")
    println("  Negative log-likelihood = $(Optim.minimum(result_q3))")


    println()
    println("================================================")


    return nothing
end


# ============================================================
# Run the program
# ============================================================

run_ps3()