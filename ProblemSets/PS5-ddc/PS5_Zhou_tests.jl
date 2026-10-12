
# PS5 - Question 4
# Unit tests

using Test
using LinearAlgebra
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM

include("PS5_Zhou_source.jl")

include("create_grids.jl")

@testset "Markov Transition Matrix" begin
    zval, zbin, xval, xbin, xtran = create_grids()

    # Check matrix dimensions
    @test size(xtran) == (zbin * xbin, xbin)

    # Check all probabilities are between 0 and 1
    @test all(xtran .>= 0.0)
    @test all(xtran .<= 1.0)
    # Check each row sums to 1
    @test all(isapprox.(sum(xtran, dims=2), 1.0; atol=1e-8))
end

@testset "Data Loading Tests" begin
    # Question 1: Static data
    df_long = load_static_data()
    @test size(df_long) == (20000, 6)
    @test length(unique(df_long.bus_id)) == 1000
    @test all(combine(groupby(df_long, :bus_id), nrow => :count).count .== 20)

    # Question 3(a): Dynamic data
    d = load_dynamic_data()
    @test size(d.Y) == (1000, 20)
    @test size(d.X) == (1000, 20)
    @test size(d.Xstate) == (1000, 20)
    @test length(d.Zstate) == 1000
end

@testset "Static Logit Model Tests" begin
    df_long = load_static_data()
    model = estimate_static_model(df_long)

    @test length(coef(model)) == 3
    @test all(isfinite, coef(model))
    @test coef(model)[2] < 0
end


@testset "Future Value Function Tests" begin
    d = load_dynamic_data()
    θ = [1.92596, -0.148154, 1.05919]

    FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
    compute_future_value!(FV, θ, d)

    @test size(FV) == (20301, 2, 21)
    @test all(FV[:, :, end] .== 0)
    @test all(isfinite, FV)
    @test any(FV[:, :, 1] .!= 0)
    # Verify Bellman recursion in the final decision period
    b = 0
    x = 1
    z = 1
    row = x + (z - 1) * d.xbin

    flow = θ[1] + θ[2] * d.xval[x] + θ[3] * b
    expected_FV = d.β * log(1 + exp(flow))

    @test isapprox(FV[row, b+1, d.T], expected_FV; atol=1e-8)

end


@testset "Log Likelihood Tests" begin
    d = load_dynamic_data()
    θ = [1.92596, -0.148154, 1.05919]

    FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
    compute_future_value!(FV, θ, d)

    loglike = compute_log_likelihood(θ, d, FV)
    neg_loglike = negative_log_likelihood(θ, d)

    @test isfinite(loglike)
    @test loglike <= 0
    @test isapprox(neg_loglike, -loglike; atol=1e-6)

end
