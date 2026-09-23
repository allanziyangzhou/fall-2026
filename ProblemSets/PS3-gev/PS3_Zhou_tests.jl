using Test
using Optim

include("PS3_Zhou_source.jl")


# ============================================================
# Question 1 Tests: Multinomial Logit
# ============================================================

@testset "Question 1: MNL" begin

    # Small artificial dataset
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

    theta_test = zeros(K * (J - 1) + 1)


    # --------------------------------------------------------
    # Test 1: parameter unpacking
    # --------------------------------------------------------
    beta, gamma = unpack_mnl_params(theta_test, K, J)

    @test size(beta) == (K, J - 1)
    @test gamma == 0.0


    # --------------------------------------------------------
    # Test 2: probability matrix dimensions
    # --------------------------------------------------------
    P = mnl_probabilities(theta_test, X_test, Z_test)

    @test size(P) == (2, 3)


    # --------------------------------------------------------
    # Test 3: probabilities sum to one
    # --------------------------------------------------------
    @test all(isapprox.(sum(P, dims=2), 1.0; atol=1e-10))


    # --------------------------------------------------------
    # Test 4: probabilities are between 0 and 1
    # --------------------------------------------------------
    @test all(P .>= 0.0)
    @test all(P .<= 1.0)


    # --------------------------------------------------------
    # Test 5: zero parameters imply equal probabilities
    # --------------------------------------------------------
    @test all(isapprox.(P, 1 / J; atol=1e-10))


    # --------------------------------------------------------
    # Test 6: negative log-likelihood is finite and nonnegative
    # --------------------------------------------------------
    nll = mnl_negloglik(theta_test, X_test, Z_test, y_test)

    @test isfinite(nll)
    @test nll >= 0.0
end

# ============================================================
# Question 3 Tests: Nested Logit
# ============================================================

@testset "Question 3: Nested Logit" begin

    # --------------------------------------------------------
    # Small artificial dataset
    # --------------------------------------------------------

    X_test_q3 = [
        1.0 0.0 1.0;
        2.0 1.0 0.0
    ]

    Z_test_q3 = [
        1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0;
        2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0
    ]

    y_test_q3 = [1, 8]

    K_q3 = size(X_test_q3, 2)
    J_q3 = size(Z_test_q3, 2)


    # --------------------------------------------------------
    # Parameter vector
    #
    # beta_WC      = 3 parameters
    # beta_BC      = 3 parameters
    # lambda_WC    = 1
    # lambda_BC    = 1
    # gamma        = 1
    #
    # Total = 9 parameters
    # --------------------------------------------------------

    theta_test_q3 = zeros(2 * K_q3 + 3)

    # Set both lambda parameters equal to 1.
    theta_test_q3[2 * K_q3 + 1] = 1.0
    theta_test_q3[2 * K_q3 + 2] = 1.0


    # --------------------------------------------------------
    # Test 1: parameter unpacking
    # --------------------------------------------------------

    beta_WC,
    beta_BC,
    lambda_WC,
    lambda_BC,
    gamma =
        unpack_nested_params(theta_test_q3, K_q3)

    @test length(beta_WC) == K_q3
    @test length(beta_BC) == K_q3

    @test beta_WC == zeros(K_q3)
    @test beta_BC == zeros(K_q3)

    @test lambda_WC == 1.0
    @test lambda_BC == 1.0
    @test gamma == 0.0


    # --------------------------------------------------------
    # Test 2: row_logsumexp
    # --------------------------------------------------------

    M_test = [
        0.0 0.0;
        1.0 2.0
    ]

    lse = row_logsumexp(M_test)

    expected_lse_1 = log(exp(0.0) + exp(0.0))
    expected_lse_2 = log(exp(1.0) + exp(2.0))

    @test isapprox(lse[1], expected_lse_1; atol=1e-10)
    @test isapprox(lse[2], expected_lse_2; atol=1e-10)


    # --------------------------------------------------------
    # Test 3: log-probability matrix dimensions
    # --------------------------------------------------------

    logP_q3 =
        nested_logit_logprobabilities(
            theta_test_q3,
            X_test_q3,
            Z_test_q3
        )

    @test size(logP_q3) == (2, 8)


    # --------------------------------------------------------
    # Test 4: probability matrix dimensions
    # --------------------------------------------------------

    P_q3 =
        nested_logit_probabilities(
            theta_test_q3,
            X_test_q3,
            Z_test_q3
        )

    @test size(P_q3) == (2, 8)


    # --------------------------------------------------------
    # Test 5: probabilities sum to one
    # --------------------------------------------------------

    probability_sums = sum(P_q3, dims=2)

    @test all(
        isapprox.(
            probability_sums,
            1.0;
            atol=1e-10
        )
    )


    # --------------------------------------------------------
    # Test 6: probabilities lie between zero and one
    # --------------------------------------------------------

    @test all(P_q3 .>= 0.0)
    @test all(P_q3 .<= 1.0)


    # --------------------------------------------------------
    # Test 7:
    # With beta = 0, gamma = 0, and both lambdas = 1,
    # all eight alternatives should have probability 1/8.
    # --------------------------------------------------------

    @test all(
        isapprox.(
            P_q3,
            1.0 / J_q3;
            atol=1e-10
        )
    )


    # --------------------------------------------------------
    # Test 8:
    # exp(log probability) should reproduce probability
    # --------------------------------------------------------

    @test all(
        isapprox.(
            exp.(logP_q3),
            P_q3;
            atol=1e-10
        )
    )


    # --------------------------------------------------------
    # Test 9: negative log-likelihood is finite
    # --------------------------------------------------------

    nll_q3 =
        nested_logit_negloglik(
            theta_test_q3,
            X_test_q3,
            Z_test_q3,
            y_test_q3
        )

    @test isfinite(nll_q3)
    @test nll_q3 >= 0.0


    # --------------------------------------------------------
    # Test 10:
    # Under equal probabilities, each chosen alternative
    # contributes -log(1/8) to the negative log-likelihood.
    # --------------------------------------------------------

    expected_nll_q3 =
        length(y_test_q3) * log(J_q3)

    @test isapprox(
        nll_q3,
        expected_nll_q3;
        atol=1e-10
    )

end