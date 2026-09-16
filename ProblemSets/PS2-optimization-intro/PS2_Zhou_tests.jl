using Test
using Optim
using LinearAlgebra
using GLM
using DataFrames

include("PS2_Zhou_source.jl")

@testset "Question 1" begin

    # Test f(x) at simple values
    @test f([0.0]) == -2.0
    @test f([1.0]) == -18.0

    # Test that minusf(x) = -f(x)
    @test minusf([0.0]) == -f([0.0])
    @test minusf([1.0]) == -f([1.0])
    @test minusf([-2.0]) == -f([-2.0])

    # Test the optimizer
    result = solve_q1([0.5])

    @test Optim.converged(result)
    @test Optim.minimizer(result)[1] ≈ -7.3782434 atol=1e-5
    @test f(Optim.minimizer(result)) ≈ 964.3134 atol=1e-3

end

@testset "Question 2" begin

    # Simple artificial data for testing ols()
    X_test = [1.0 1.0;
              1.0 2.0;
              1.0 3.0]

    y_test = [1.0, 2.0, 3.0]

    # Test the SSR calculation
    beta_test = [0.0, 1.0]
    @test ols(beta_test, X_test, y_test) ≈ 0.0 atol=1e-10

    beta_wrong = [1.0, 0.0]
    @test ols(beta_wrong, X_test, y_test) > 0

    # Estimate beta using Optim
    result_q2 = optimize(
        b -> ols(b, X_test, y_test),
        zeros(size(X_test, 2)),
        LBFGS()
    )

    @test Optim.converged(result_q2)

    beta_optim = Optim.minimizer(result_q2)

    # Compare with closed-form OLS
    beta_matrix = inv(X_test' * X_test) * X_test' * y_test

    @test beta_optim ≈ beta_matrix atol=1e-6

    # Expected coefficients are approximately [0, 1]
    @test beta_optim[1] ≈ 0.0 atol=1e-6
    @test beta_optim[2] ≈ 1.0 atol=1e-6

end

@testset "Question 3 and 4" begin

    X_test = [1.0 0.0;
              1.0 1.0;
              1.0 2.0;
              1.0 3.0;
              1.0 4.0;
              1.0 5.0]

    d_test = [0, 1, 0, 1, 0, 1]

    alpha_test = [0.0, 0.0]
    ll = logit(alpha_test, X_test, d_test)

    @test isfinite(ll)
    @test ll ≈ length(d_test) * log(0.5) atol=1e-10

    result_logit = optimize(
        a -> -logit(a, X_test, d_test),
        zeros(size(X_test, 2)),
        LBFGS()
    )

    @test Optim.converged(result_logit)

    beta_optim = Optim.minimizer(result_logit)

df_test = DataFrame(
    d = d_test,
    x = X_test[:, 2]
)

logit_glm_test = glm(
    @formula(d ~ x),
    df_test,
    Binomial(),
    LogitLink()
)

beta_glm = coef(logit_glm_test)

@test beta_optim ≈ beta_glm atol=1e-4

end

@testset "Question 5" begin

    X_test = [1.0 0.0;
              1.0 1.0;
              1.0 2.0;
              1.0 3.0]

    d_test = [1, 2, 7, 1]

    alpha0 = zeros(size(X_test, 2) * 6)

    ll = mlogit(alpha0, X_test, d_test)

    @test isfinite(ll)

    result_q5_test = optimize(
        a -> -mlogit(a, X_test, d_test),
        alpha0,
        LBFGS(),
        Optim.Options(g_tol = 1e-5)
    )

    @test Optim.converged(result_q5_test)

    beta_q5_test = reshape(
        Optim.minimizer(result_q5_test),
        size(X_test, 2),
        6
    )

    @test size(beta_q5_test) == (2, 6)

end