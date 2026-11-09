
# PS8 - Main Script

using Optim, HTTP, GLM, LinearAlgebra, Random
using Statistics, DataFrames, DataFramesMeta, CSV
using MultivariateStats, Distributions, FreqTables
using ForwardDiff, LineSearches

include("lgwt.jl")
include("PS8_Zhou_source.jl")

function main()

    # Question 1: Baseline Wage Regression

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS8-factor/nlsy.csv"

    df = load_data(url)

    println("Question 1: Baseline Wage Regression")
    println("Data dimensions: ", size(df))

    model = estimate_base_regression(df)

    println("Regression Results:")
    display(coeftable(model))   

    # Question 2: ASVAB Correlations

    println("\nQuestion 2: ASVAB Correlations")

    corr_matrix = compute_asvab_correlations(df)

    println("ASVAB Correlation Matrix:")
    display(corr_matrix)

    # Question 3: Wage Regression with Six ASVAB Scores

    println("\nQuestion 3: Full Wage Regression with ASVAB Scores")

    model_full = estimate_full_regression(df)

    println("Regression Results:")
    display(coeftable(model_full))

    # Q3 Discussion:
    # The six ASVAB scores are moderately to strongly correlated,
    # with pairwise correlations ranging from about 0.42 to 0.80.
    # Including all six scores may cause multicollinearity, making
    # individual coefficients less precise and harder to interpret.
    # In the full regression, only asvabAR and asvabMK are significant
    # at the 5% level.
    # Compared with Q1, the coefficients on black and hispanic
    # move closer to zero, and hispanic becomes insignificant.
    # Multicollinearity does not necessarily bias OLS estimates,
    # but it can increase their standard errors.


    # Question 4: Principal Component Analysis (PCA)

    println("\nQuestion 4: PCA Wage Regression")

    model_pca = estimate_pca_regression(df)

    println("Regression Results with First Principal Component:")
    display(coeftable(model_pca))

    # Question 5: Factor Analysis Regression

    println("\nQuestion 5: Factor Analysis Wage Regression")

    model_factor = estimate_factor_regression(df)

    println("Regression Results with One Factor:")
    display(coeftable(model_factor))

    # Question 6(a): Prepare Factor Model Data

    println("\nQuestion 6: Full Factor Model (MLE)")

    X, y, Xfac, asvabs = prepare_factor_matrices(df)

    println("X dimensions: ", size(X))
    println("y length: ", length(y))
    println("Xfac dimensions: ", size(Xfac))
    println("ASVAB dimensions: ", size(asvabs))
    
    # Question 6: Test the negative log-likelihood

    nodes, weights = lgwt(40, -8, 8)

    theta0 = initialize_factor_parameters(X, y, Xfac, asvabs)

    nll0 = factor_negloglik(
        theta0, X, y, Xfac, asvabs, nodes, weights
    )

    println("Initial negative log-likelihood: ", nll0)
    println("Likelihood is finite: ", isfinite(nll0))
    println("Number of parameters: ", length(theta0))
    println("Initial parameters are finite: ", all(isfinite, theta0))
    
# Question 6: Small-sample MLE optimization test

N_test = 100
nodes_test, weights_test = lgwt(10, -8, 8)

objective_test = theta -> factor_negloglik(
    theta,
    X[1:N_test, :],
    y[1:N_test],
    Xfac[1:N_test, :],
    asvabs[1:N_test, :],
    nodes_test,
    weights_test
)

result_test = optimize(
    objective_test,
    theta0,
    LBFGS(),
    Optim.Options(iterations=200)
)

println("Small-sample MLE objective: ", Optim.minimum(result_test))
println("Optimization converged: ", Optim.converged(result_test))

# Question 6: Full-sample MLE estimation

objective_full = theta -> factor_negloglik(
    theta, X, y, Xfac, asvabs, nodes, weights
)

result_full = optimize(
    objective_full,
    theta0,
    LBFGS(),
    Optim.Options(iterations=300)
)

println("\nFull-sample MLE Results:")
println("Number of observations: ", length(y))
println("Number of parameters: ", length(Optim.minimizer(result_full)))
println("Negative log-likelihood: ", Optim.minimum(result_full))
println("Optimization converged: ", Optim.converged(result_full))

# Display the estimated parameters

alpha_hat, beta_hat, gamma_hat, delta_hat, sigma_hat =
    unpack_factor_parameters(Optim.minimizer(result_full))

println("\nMeasurement coefficients (alpha):")
display(alpha_hat)

println("Wage coefficients (beta): ", beta_hat)
println("ASVAB factor loadings (gamma): ", gamma_hat)
println("Wage factor loading (delta): ", delta_hat)
println("Error standard deviations (sigma): ", sigma_hat)

    return nothing
end

@time main()
