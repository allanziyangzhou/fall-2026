# ============================================================
# PS4_Zhou_tests.jl
# ECON 6343 - Problem Set 4
# ============================================================


# ============================================================
# Load packages
# ============================================================

using Test
using Optim
using LinearAlgebra
using Random
using ForwardDiff
using ADTypes: AutoForwardDiff
using Distributions


# ============================================================
# Load functions
# ============================================================

include(joinpath(@__DIR__, "lgwt.jl"))
include(joinpath(@__DIR__, "PS4_Zhou_source.jl"))


# ============================================================
# Question 1 Tests:
# Multinomial Logit with Alternative-Specific Z
# ============================================================

@testset "Question 1: Multinomial Logit" begin

    # --------------------------------------------------------
    # Small artificial dataset
    # --------------------------------------------------------

    X_test = [
        1.0 0.0 1.0;
        2.0 1.0 0.0
    ]

    Z_test = [
        1.0 2.0 3.0;
        2.0 1.0 3.0
    ]

    y_test = [1, 2]

    K = size(X_test, 2)
    J = size(Z_test, 2)

    theta_test =
        zeros(
            K * (J - 1) + 1
        )


    # --------------------------------------------------------
    # Test 1:
    # Parameter unpacking
    # --------------------------------------------------------

    alpha_test,
    gamma_test =
        unpack_mlogit_params(
            theta_test,
            K,
            J
        )

    @test size(alpha_test) == (K, J - 1)

    @test gamma_test == 0.0


    # --------------------------------------------------------
    # Test 2:
    # Probability matrix dimensions
    # --------------------------------------------------------

    P_test =
        mlogit_probabilities(
            theta_test,
            X_test,
            Z_test
        )

    @test size(P_test) == (2, 3)


    # --------------------------------------------------------
    # Test 3:
    # Probabilities sum to one
    # --------------------------------------------------------

    @test all(
        isapprox.(
            sum(P_test, dims = 2),
            1.0;
            atol = 1e-10
        )
    )


    # --------------------------------------------------------
    # Test 4:
    # Probabilities lie between zero and one
    # --------------------------------------------------------

    @test all(P_test .>= 0.0)

    @test all(P_test .<= 1.0)


    # --------------------------------------------------------
    # Test 5:
    # Zero parameters imply equal probabilities
    # --------------------------------------------------------

    @test all(
        isapprox.(
            P_test,
            1.0 / J;
            atol = 1e-10
        )
    )


    # --------------------------------------------------------
    # Test 6:
    # Negative log-likelihood is finite
    # --------------------------------------------------------

    nll_test =
        mlogit_with_Z(
            theta_test,
            X_test,
            Z_test,
            y_test
        )

    @test isfinite(nll_test)

    @test nll_test >= 0.0


    # --------------------------------------------------------
    # Test 7:
    # With equal probabilities,
    # NLL should equal N * log(J)
    # --------------------------------------------------------

    expected_nll =
        length(y_test) * log(J)

    @test isapprox(
        nll_test,
        expected_nll;
        atol = 1e-10
    )

end


# ============================================================
# Question 3(a) Tests:
# Quadrature with Normal(0,1)
# ============================================================

@testset "Question 3(a): Normal Quadrature Check" begin

    density_integral,
    mean_integral =
        quadrature_normal_check()


    # Integral of density should be close to 1.
    # Seven quadrature points give an approximation.
    @test isapprox(
        density_integral,
        1.0;
        atol = 0.01
    )


    # Symmetry implies the mean should be approximately zero.
    @test isapprox(
        mean_integral,
        0.0;
        atol = 1e-10
    )

end


# ============================================================
# Question 3(b) Tests:
# Variance using quadrature
# ============================================================

@testset "Question 3(b): Quadrature Variance" begin

    variance_7 =
        quadrature_variance(7)

    variance_10 =
        quadrature_variance(10)


    # Both approximations should be finite and positive.
    @test isfinite(variance_7)

    @test isfinite(variance_10)

    @test variance_7 > 0.0

    @test variance_10 > 0.0


    # True variance of Normal(0,2) is 4.
    # Ten quadrature points should provide a reasonably
    # accurate approximation.
    @test isapprox(
        variance_10,
        4.0;
        atol = 0.1
    )


    # Ten points should improve on seven points.
    @test abs(variance_10 - 4.0) <
          abs(variance_7 - 4.0)

end


# ============================================================
# Question 3(c) Tests:
# Monte Carlo Integration
# ============================================================

@testset "Question 3(c): Monte Carlo Integration" begin

    mc_test =
        monte_carlo_integrals(
            1_000_000;
            seed = 1234
        )


    # --------------------------------------------------------
    # Integral x^2 f(x) should be approximately 4
    # --------------------------------------------------------

    @test isapprox(
        mc_test.second_moment,
        4.0;
        atol = 0.1
    )


    # --------------------------------------------------------
    # Integral x f(x) should be approximately 0
    # --------------------------------------------------------

    @test isapprox(
        mc_test.first_moment,
        0.0;
        atol = 0.05
    )


    # --------------------------------------------------------
    # Integral f(x) should be approximately 1
    # --------------------------------------------------------

    @test isapprox(
        mc_test.density_integral,
        1.0;
        atol = 0.02
    )

end


# ============================================================
# Question 4 Tests:
# Mixed Logit with Quadrature
#
# We test the likelihood function only.
# The full estimation is intentionally NOT run.
# ============================================================

@testset "Question 4: Mixed Logit Quadrature" begin

    # --------------------------------------------------------
    # Small artificial dataset
    # --------------------------------------------------------

    X_test_q4 = [
        1.0 0.0 1.0;
        2.0 1.0 0.0
    ]

    Z_test_q4 = [
        1.0 2.0 3.0;
        2.0 1.0 3.0
    ]

    y_test_q4 = [1, 2]

    K_q4 = size(X_test_q4, 2)
    J_q4 = size(Z_test_q4, 2)


    # --------------------------------------------------------
    # Parameter vector:
    #
    # K*(J-1) alpha parameters
    # + mu_gamma
    # + sigma_gamma
    # --------------------------------------------------------

    theta_test_q4 =
        zeros(
            K_q4 * (J_q4 - 1) + 2
        )

    # sigma_gamma must be positive
    theta_test_q4[end] = 1.0


    # --------------------------------------------------------
    # Quadrature nodes and weights
    # --------------------------------------------------------

    nodes_test_q4,
    weights_test_q4 =
        lgwt(
            7,
            -4,
            4
        )


    # --------------------------------------------------------
    # Test 1:
    # Correct number of nodes and weights
    # --------------------------------------------------------

    @test length(nodes_test_q4) == 7

    @test length(weights_test_q4) == 7


    # --------------------------------------------------------
    # Test 2:
    # Weights should be positive
    # --------------------------------------------------------

    @test all(weights_test_q4 .> 0.0)


    # --------------------------------------------------------
    # Test 3:
    # Mixed-logit negative log-likelihood should be finite
    # --------------------------------------------------------

    nll_q4 =
        mixed_logit_quad(
            theta_test_q4,
            X_test_q4,
            Z_test_q4,
            y_test_q4,
            nodes_test_q4,
            weights_test_q4
        )

    @test isfinite(nll_q4)

    @test nll_q4 >= 0.0

end