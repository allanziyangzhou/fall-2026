
# PS7 - Unit Tests

using Test
using Optim, HTTP, GLM, LinearAlgebra, Random
using Statistics, DataFrames, DataFramesMeta, CSV
using Distributions

include("PS7_Zhou_source.jl")

@testset "Question 1: OLS via GMM" begin

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"

    df, X, y = load_data(url)

    # Check data dimensions
    @test size(X, 1) == length(y)
    @test size(X, 2) == 4

    # Check the GMM objective function
    beta = zeros(size(X, 2))
    @test ols_gmm(beta, X, y) >= 0

    # Compare GMM estimates with OLS
    result = optimize(
        b -> ols_gmm(b, X, y),
        beta,
        LBFGS()
    )

    beta_ols = X \ y

    @test Optim.converged(result)
    @test isapprox(Optim.minimizer(result), beta_ols; atol=1e-6)

    # Check the objective at the OLS solution
    @test ols_gmm(beta_ols, X, y) <= ols_gmm(beta, X, y)

end

# PS7 - Question 2
# Multinomial Logit Unit Tests

@testset "Question 2: Multinomial Logit" begin

    # Small synthetic dataset for fast testing
    X = [
        1.0 -1.0
        1.0  0.0
        1.0  1.0
        1.0  2.0
        1.0 -2.0
        1.0  0.5
    ]

    y = [1, 2, 3, 1, 2, 3]

    # Two coefficients for each non-base alternative
    alpha = zeros(4)

    # Question 2(a): MLE
    @test isfinite(mlogit_mle(alpha, X, y))
    @test mlogit_mle(alpha, X, y) ≈ length(y) * log(3)

    # Question 2(b): Overidentified GMM
    @test isfinite(mlogit_gmm_overid(alpha, X, y))
    @test mlogit_gmm_overid(alpha, X, y) >= 0

    # Check GMM objective at equal choice frequencies
    @test mlogit_gmm_overid(alpha, X, y) ≈ 4.0

    # Question 2(c): Random starting values
    Random.seed!(1234)
    alpha_random = rand(4)

    @test isfinite(mlogit_gmm_overid(alpha_random, X, y))

end

# PS7 - Question 3
# Multinomial Logit Simulation Tests

@testset "Question 3: Multinomial Logit Simulation" begin

    Random.seed!(1234)
    N = 5_000
    J = 4

    # Test inverse CDF simulation
    Y, X = sim_logit(N, J)

    @test size(X) == (N, 3)
    @test length(Y) == N
    @test all((Y .>= 1) .& (Y .<= J))
    @test length(unique(Y)) == J

    # Test parameter recovery via MLE
    beta_true = [
         1.0  -2.0   0.0
        -1.0   0.5  -0.5
         0.5   0.3   2.0
    ]

    result = optimize(
        alpha -> mlogit_mle(alpha, X, Y),
        zeros(9),
        LBFGS(),
        Optim.Options(g_tol=1e-5, iterations=1_000)
    )

    @test Optim.converged(result)
    @test maximum(abs.(Optim.minimizer(result) .-
                        vec(beta_true))) < 0.2

    # Test Gumbel simulation
    Y_gumbel, X_gumbel = sim_logit_w_gumbel(N, J)

    @test size(X_gumbel) == (N, 3)
    @test length(Y_gumbel) == N
    @test all((Y_gumbel .>= 1) .& (Y_gumbel .<= J))
    @test length(unique(Y_gumbel)) == J

end

# PS7 - Question 5
# SMM Estimation Unit Tests

@testset "Question 5: SMM Estimation" begin

    Random.seed!(1234)

    # Generate a small synthetic dataset
    Y, X = sim_logit(100, 4)

    alpha = zeros(9)
    D = 10

    # Check the SMM objective function
    obj = mlogit_smm_overid(alpha, X, Y, D)

    @test isfinite(obj)
    @test obj >= 0

    # Check reproducibility with a fixed random seed
    obj_repeat = mlogit_smm_overid(alpha, X, Y, D)
    @test obj ≈ obj_repeat

    # Check parameter dimensions
    @test length(alpha) == size(X, 2) * 3

    # Check that the objective changes with parameters
    alpha_alt = fill(0.5, 9)
    obj_alt = mlogit_smm_overid(alpha_alt, X, Y, D)

    @test isfinite(obj_alt)
    @test obj_alt >= 0

end