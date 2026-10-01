# ============================================================
# PS4_Zhou_script.jl
# ECON 6343 - Problem Set 4
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
using ForwardDiff
using ADTypes: AutoForwardDiff
using Distributions

include(joinpath(@__DIR__, "lgwt.jl"))
include(joinpath(@__DIR__, "PS4_Zhou_source.jl"))

# ============================================================
# Load function definitions
# ============================================================
# Main function
# ============================================================

function run_ps4()

    # ========================================================
    # Load data
    # ========================================================

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS4-mixture/nlsw88t.csv"

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

    y = df.occ_code


    # ========================================================
    # Names for printing
    # ========================================================

    x_names = [
        "age",
        "white",
        "collgrad"
    ]

    occupation_names = [
        "Professional/Technical",
        "Managers/Administrators",
        "Sales",
        "Clerical/Unskilled",
        "Craftsmen",
        "Operatives",
        "Transport"
    ]


    # ========================================================
    # Question 1:
    # Multinomial Logit with Alternative-Specific Covariates
    # ========================================================

    K = size(X, 2)
    J = size(Z, 2)


    # --------------------------------------------------------
    # Starting values
    # --------------------------------------------------------

    theta0 = zeros(K * (J - 1) + 1)


    # --------------------------------------------------------
    # Estimate model
    # --------------------------------------------------------

    q1 = estimate_mlogit_with_Z(
        X,
        Z,
        y;
        theta0 = theta0
    )


    # --------------------------------------------------------
    # Extract results
    # --------------------------------------------------------

    result_q1 = q1.result

    theta_hat_q1 = q1.theta_hat

    alpha_hat_q1 = q1.alpha_hat
    gamma_hat_q1 = q1.gamma_hat

    alpha_se_q1 = q1.alpha_se
    gamma_se_q1 = q1.gamma_se


    # --------------------------------------------------------
    # Print Question 1 results
    # --------------------------------------------------------

    println()
    println("================================================")
    println("Question 1: Multinomial Logit")
    println("================================================")


    for j in 1:(J - 1)

        println()
        println("Alternative $j: $(occupation_names[j])")

        for k in 1:K

            println(
                "  $(x_names[k]) = ",
                alpha_hat_q1[k, j],
                "   SE = ",
                alpha_se_q1[k, j]
            )

        end

    end


    # --------------------------------------------------------
    # Normalized alternative
    # --------------------------------------------------------

    println()
    println("Alternative 8: Other")
    println("  beta = 0 (normalized)")


    # --------------------------------------------------------
    # Gamma
    # --------------------------------------------------------

    println()
    println("Estimated gamma:")
    println(
        "  gamma = ",
        gamma_hat_q1,
        "   SE = ",
        gamma_se_q1
    )


    # --------------------------------------------------------
    # Optimization information
    # --------------------------------------------------------

    println()
    println("Optimization:")
    println(
        "  Converged = ",
        Optim.converged(result_q1)
    )

    println(
        "  Negative log-likelihood = ",
        Optim.minimum(result_q1)
    )


    println()
    println("================================================")

    
    # ========================================================
    # Question 2
    # ========================================================

    # Yes. The estimated gamma is positive (about 1.31), so a higher
    # expected wage for an occupation increases the likelihood of
    # choosing that occupation. This is more intuitive than the
    # negative gamma estimated in Problem Set 3.

    # ========================================================
    # Question 3(a): Quadrature practice
    # ========================================================

    density_q3a,
    mean_q3a =
        quadrature_normal_check()

    println()
    println("================================================")
    println("Question 3(a): Quadrature with Normal(0,1)")
    println("================================================")

    println(
        "Integral of density = ",
        density_q3a
    )

    println(
        "Integral of x*f(x) = ",
        mean_q3a
    )


    # ========================================================
    # Question 3(b): Variance using quadrature
    # ========================================================

    variance_7 =
        quadrature_variance(7)

    variance_10 =
        quadrature_variance(10)

    println()
    println("================================================")
    println("Question 3(b): Quadrature Variance")
    println("================================================")

    println(
        "7 quadrature points  = ",
        variance_7
    )

    println(
        "10 quadrature points = ",
        variance_10
    )

    println(
        "True variance        = 4.0"
    )


    # ========================================================
    # Question 3(c): Monte Carlo integration
    # ========================================================

    mc_1000 =
        monte_carlo_integrals(1000)

    mc_1000000 =
        monte_carlo_integrals(1_000_000)

    println()
    println("================================================")
    println("Question 3(c): Monte Carlo Integration")
    println("================================================")

    println()
    println("D = 1,000")

    println(
        "Integral x^2*f(x) = ",
        mc_1000.second_moment
    )

    println(
        "Integral x*f(x)   = ",
        mc_1000.first_moment
    )

    println(
        "Integral f(x)     = ",
        mc_1000.density_integral
    )


    println()
    println("D = 1,000,000")

    println(
        "Integral x^2*f(x) = ",
        mc_1000000.second_moment
    )

    println(
        "Integral x*f(x)   = ",
        mc_1000000.first_moment
    )

    println(
        "Integral f(x)     = ",
        mc_1000000.density_integral
    )


    # ========================================================
    # Question 3(d)
    # ========================================================

    # Quadrature uses predetermined nodes and different weights,
    # while Monte Carlo uses random nodes drawn uniformly and the
    # same weight, (b-a)/D, for every draw. Quadrature can provide
    # a very accurate approximation with relatively few points,
    # while Monte Carlo generally requires many more draws.

    # ========================================================
    # Question 4: Mixed Logit with Quadrature
    # ========================================================

    # gamma ~ N(mu_gamma, sigma_gamma^2)
    # The integral is approximated using Gauss-Legendre quadrature.
    # Following the problem-set instructions, this model is set up
    # but not actually estimated.


    # --------------------------------------------------------
    # Quadrature nodes and weights
    # --------------------------------------------------------

    nodes_q4,
    weights_q4 =
        lgwt(
            7,
            -4,
            4
        )


    # --------------------------------------------------------
    # Starting values
    #
    # 21 alpha parameters
    # + mu_gamma
    # + sigma_gamma
    # --------------------------------------------------------

    theta0_q4 =
        zeros(
            K * (J - 1) + 2
        )

    theta0_q4[end] = 1.0


    # --------------------------------------------------------
    # DO NOT RUN
    # --------------------------------------------------------

    # result_q4,
    # theta_hat_q4 =
    #     estimate_mixed_logit_quad(
    #         X,
    #         Z,
    #         y,
    #         nodes_q4,
    #         weights_q4;
    #         theta0 = theta0_q4
    #     )

        # ========================================================
    # Question 6
    # ========================================================

    return nothing
end


# ============================================================
# Run the program
# ============================================================

run_ps4()