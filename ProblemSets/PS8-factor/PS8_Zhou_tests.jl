
# PS8 - Unit Tests

using Test
using Optim, HTTP, GLM, LinearAlgebra, Random
using Statistics, DataFrames, DataFramesMeta, CSV
using MultivariateStats, Distributions, FreqTables
using ForwardDiff, LineSearches

include("lgwt.jl")
include("PS8_Zhou_source.jl")

# Question 1: Baseline Wage Regression
@testset "Question 1: Baseline Wage Regression" begin

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS8-factor/nlsy.csv"

    df = load_data(url)
    model = estimate_base_regression(df)

    # Check data loading
    @test df isa DataFrame
    @test nrow(df) > 0
    @test all(in.(["logwage", "black", "hispanic", "female",
                   "schoolt", "gradHS", "grad4yr"], Ref(names(df))))

    # Check regression estimates
    @test length(coef(model)) == 7
    @test all(isfinite, coef(model))
    @test all(isfinite, stderror(model))
    @test nobs(model) > 0

end

# Question 2: ASVAB Correlations
@testset "Question 2: ASVAB Correlations" begin

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS8-factor/nlsy.csv"
    df = load_data(url)

    C = compute_asvab_correlations(df)

    @test size(C) == (6, 6)
    @test all(isfinite, C)
    @test isapprox(C, C'; atol=1e-10)
    @test all(isapprox.(diag(C), 1.0; atol=1e-10))
    @test all(abs.(C) .<= 1.0 + 1e-10)
    @test minimum(C) >= -1.0 - 1e-10

end

# Question 3: Full Wage Regression
@testset "Question 3: Full Wage Regression" begin

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS8-factor/nlsy.csv"
    df = load_data(url)

    model_full = estimate_full_regression(df)

    # Check all 12 regressors and the intercept
    @test length(coef(model_full)) == 13
    @test all(isfinite, coef(model_full))
    @test all(isfinite, stderror(model_full))
    @test nobs(model_full) > 0

    # Check the six ASVAB variables are included
    asvab_names = ["asvabAR", "asvabCS", "asvabMK",
                   "asvabNO", "asvabPC", "asvabWK"]
    @test all(in.(asvab_names, Ref(coefnames(model_full))))

    # Check the model fits the same sample as Q1
    model_base = estimate_base_regression(df)
    @test nobs(model_full) == nobs(model_base)

end

# Question 4: PCA Wage Regression
@testset "Question 4: PCA Wage Regression" begin

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS8-factor/nlsy.csv"
    df = load_data(url)

    model_pca = estimate_pca_regression(df)

    @test length(coef(model_pca)) == 8
    @test all(isfinite, coef(model_pca))
    @test all(isfinite, stderror(model_pca))
    @test nobs(model_pca) == nrow(df)
    @test "asvabPCA" in coefnames(model_pca)

    # Verify the first principal component has one score per observation
    asvab_cols = [:asvabAR, :asvabCS, :asvabMK,
                  :asvabNO, :asvabPC, :asvabWK]

    X_pca = permutedims(Matrix{Float64}(df[:, asvab_cols]))
    pca = fit(PCA, X_pca; maxoutdim=1)
    scores = MultivariateStats.transform(pca, X_pca)

    @test size(scores) == (1, nrow(df))
    @test all(isfinite, scores)

end

# Question 5: Factor Analysis Regression
@testset "Question 5: Factor Analysis Regression" begin

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS8-factor/nlsy.csv"
    df = load_data(url)

    model_factor = estimate_factor_regression(df)

    @test length(coef(model_factor)) == 8
    @test all(isfinite, coef(model_factor))
    @test all(isfinite, stderror(model_factor))
    @test nobs(model_factor) == nrow(df)
    @test "asvabFactor" in coefnames(model_factor)

    # Verify that one factor score is generated for each observation
    asvab_cols = [:asvabAR, :asvabCS, :asvabMK,
                  :asvabNO, :asvabPC, :asvabWK]

    X_factor = permutedims(Matrix{Float64}(df[:, asvab_cols]))
    factor = fit(FactorAnalysis, X_factor; maxoutdim=1)
    scores = MultivariateStats.transform(factor, X_factor)

    @test size(scores) == (1, nrow(df))
    @test all(isfinite, scores)

end

# Question 6: Full Factor Model
@testset "Question 6: Full Factor Model" begin

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS8-factor/nlsy.csv"
    df = load_data(url)

    X, y, Xfac, asvabs = prepare_factor_matrices(df)

    @test size(X) == (nrow(df), 7)
    @test length(y) == nrow(df)
    @test size(Xfac) == (nrow(df), 4)
    @test size(asvabs) == (nrow(df), 6)

    theta0 = initialize_factor_parameters(X, y, Xfac, asvabs)

    @test length(theta0) == 45
    @test all(isfinite, theta0)

    alpha, beta, gamma, delta, sigma =
        unpack_factor_parameters(theta0)

    @test size(alpha) == (4, 6)
    @test length(beta) == 7
    @test length(gamma) == 6
    @test delta isa Real
    @test length(sigma) == 7
    @test all(sigma .> 0)

    nodes, weights = lgwt(20, -8, 8)
    nll = factor_negloglik(theta0, X, y, Xfac, asvabs, nodes, weights)

    @test isfinite(nll)
    @test nll > 0

end
