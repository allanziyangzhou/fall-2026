
# PS6 - Question 4
# Unit Tests for Question 1

using Test
using HTTP, CSV, DataFrames, DataFramesMeta
using LinearAlgebra

include("create_grids.jl")

include("PS6_Zhou_source.jl")

@testset "Question 1: Data Loading" begin

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS5-ddc/busdata.csv"

    df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)

    # Check data dimensions
    @test nrow(df_long) == 20000
    @test ncol(df_long) == 6

    # Check panel structure
    @test length(unique(df_long.bus_id)) == 1000
    @test all(combine(groupby(df_long, :bus_id),
                      nrow => :count).count .== 20)

    # Check observation ordering
    @test issorted(df_long, [:bus_id, :time])

    # Check state dimensions
    @test size(Xstate) == (1000, 20)
    @test length(Zstate) == 1000
    @test length(Branded) == 1000

end


# PS6 - Question 2
# Flexible Logit Model Tests

@testset "Question 2: Flexible Logit" begin

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS5-ddc/busdata.csv"

    df_long, _, _, _ = load_and_reshape_data(url)

    model = estimate_flexible_logit(df_long)

    # Check coefficient estimates
    @test length(coef(model)) == 128
    @test all(isfinite, coef(model))

    # Check log likelihood
    @test isfinite(loglikelihood(model))
    @test loglikelihood(model) <= 0

    # Check predicted probabilities
    p = predict(model)

    @test length(p) == nrow(df_long)
    @test all(isfinite, p)
    @test all((p .>= 0.0) .& (p .<= 1.0))

end


# PS6 - Question 3(a)
# State Transition Matrix Tests

@testset "Question 3(a): Transition Matrix" begin

    zval, zbin, xval, xbin, xtran = create_grids()

    @test zbin == 101
    @test xbin == 201
    @test size(xtran) == (zbin * xbin, xbin)

    # All transition probabilities must be valid
    @test all(xtran .>= 0.0)
    @test all(xtran .<= 1.0)

    # Each row must sum to one
    @test all(isapprox.(sum(xtran, dims=2), 1.0; atol=1e-8))

end


# PS6 - Question 3(b)
# State Space Tests

@testset "Question 3(b): State Space" begin

    zval, zbin, xval, xbin, xtran = create_grids()
    state_df = construct_state_space(xbin, zbin, xval, zval)

    @test size(state_df) == (zbin * xbin, 4)
    @test names(state_df) == ["Odometer", "RouteUsage", "Branded", "time"]

    # Verify state ordering matches the transition matrix
    @test state_df.Odometer == kron(ones(zbin), xval)
    @test state_df.RouteUsage == kron(zval, ones(xbin))

    # Verify initial values
    @test all(state_df.Branded .== 0)
    @test all(state_df.time .== 0)

end


# PS6 - Question 3(b)
# CCP Future Value Tests

@testset "Question 3(b): Future Values" begin
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS5-ddc/busdata.csv"
    df_long, Xstate, Zstate, Branded =
        load_and_reshape_data(url)

    flex_logit = estimate_flexible_logit(df_long)

    zval, zbin, xval, xbin, xtran = create_grids()
    state_df = construct_state_space(xbin, zbin, xval, zval)

    T = 20
    β = 0.9

    FV = compute_future_values(
        state_df, flex_logit, xtran, xbin, T, β
    )

    @test size(FV) == (xbin * zbin, 2, T + 1)
    @test all(isfinite, FV)
    @test all(FV .>= 0)
    @test all(FV[:, :, 1] .== 0)
    @test all(FV[:, :, end] .== 0)

    # Check the CCP formula for one state
    state_df.time .= 2
    state_df.Branded .= 0

    p0 = clamp(
        1.0 - predict(flex_logit, state_df)[1],
        eps(Float64), 1.0
    )

    @test FV[1, 1, 2] ≈ -β * log(p0)

end


# PS6 - Question 3(b)
# FVT1 Tests

@testset "Question 3(b): FVT1" begin

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS5-ddc/busdata.csv"

    df_long, Xstate, Zstate, Branded =
        load_and_reshape_data(url)

    flex_logit = estimate_flexible_logit(df_long)

    zval, zbin, xval, xbin, xtran = create_grids()
    state_df = construct_state_space(xbin, zbin, xval, zval)

    FV = compute_future_values(
        state_df, flex_logit, xtran, xbin, 20, 0.9
    )

    fvt1 = compute_fvt1(
        df_long, FV, xtran, Xstate, Zstate, xbin, Branded
    )

    @test length(fvt1) == nrow(df_long)
    @test all(isfinite, fvt1)

    # Verify the first observation manually
    row0 = 1 + (Zstate[1] - 1) * xbin
    row1 = Xstate[1, 1] + (Zstate[1] - 1) * xbin

    expected = dot(
        xtran[row1, :] - xtran[row0, :],
        FV[row0:(row0 + xbin - 1), Branded[1] + 1, 2]
    )

    @test fvt1[1] ≈ expected

    # Verify future value differences in the terminal period
    @test all(fvt1[20:20:end] .≈ 0.0)

end


# PS6 - Question 3(c)
# Structural Parameter Estimation Tests

@testset "Question 3(c): Structural Parameters" begin

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS5-ddc/busdata.csv"

    df_long, Xstate, Zstate, Branded =
        load_and_reshape_data(url)

    flex_logit = estimate_flexible_logit(df_long)

    zval, zbin, xval, xbin, xtran = create_grids()
    state_df = construct_state_space(xbin, zbin, xval, zval)

    FV = compute_future_values(
        state_df, flex_logit, xtran, xbin, 20, 0.9
    )

    fvt1 = compute_fvt1(
        df_long, FV, xtran, Xstate, Zstate, xbin, Branded
    )

    structural_logit = estimate_structural_params(df_long, fvt1)

    # Check coefficient estimates
    @test length(coef(structural_logit)) == 3
    @test all(isfinite, coef(structural_logit))

    # Check model fit
    @test isfinite(loglikelihood(structural_logit))
    @test loglikelihood(structural_logit) <= 0

    # Check fitted probabilities
    p = predict(structural_logit)
    @test all(isfinite, p)
    @test all((p .>= 0) .& (p .<= 1))

end


