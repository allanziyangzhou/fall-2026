
# PS7 - Main Script

using Optim, HTTP, GLM, LinearAlgebra, Random
using Statistics, DataFrames, DataFramesMeta, CSV
using Distributions

include("PS7_Zhou_source.jl")

function main()

    # Question 1: OLS via GMM

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"

    df, X, y = load_data(url)

    # Estimate coefficients using GMM
    result_gmm = optimize(
        beta -> ols_gmm(beta, X, y),
        zeros(size(X, 2)),
        LBFGS()
    )

    # Compute closed-form OLS estimates
    beta_ols = X \ y

    println("Question 1: OLS via GMM")
    println("GMM estimates: ", Optim.minimizer(result_gmm))
    println("OLS estimates: ", beta_ols)
    println("Optimization converged: ", Optim.converged(result_gmm))
    println("Maximum absolute difference: ",
            maximum(abs.(Optim.minimizer(result_gmm) .- beta_ols)))

    # Question 2(a): Multinomial Logit via MLE

    df_occ, X_occ, y_occ = prepare_occupation_data(df)

    # Initialize coefficients for 6 non-base alternatives
    alpha_start = zeros(size(X_occ, 2) * 6)

    result_mle = optimize(
    alpha -> mlogit_mle(alpha, X_occ, y_occ),
    alpha_start,
    LBFGS(),
    Optim.Options(g_tol=1e-5, iterations=100_000)
    )

    println("Question 2(a): Multinomial Logit via MLE")
    println("MLE estimates: ", Optim.minimizer(result_mle))
    println("Optimization converged: ", Optim.converged(result_mle))
    println("Negative log-likelihood: ", Optim.minimum(result_mle))      
    
    
# Question 2(b): GMM with MLE Starting Values

    result_gmm_mle = optimize(
    alpha -> mlogit_gmm_overid(alpha, X_occ, y_occ),
    Optim.minimizer(result_mle),
    LBFGS(),
    Optim.Options(g_tol=1e-5, iterations=100_000)
    )

    println("Question 2(b): GMM with MLE Starting Values")
    println("GMM estimates: ", Optim.minimizer(result_gmm_mle))
    println("Optimization converged: ", Optim.converged(result_gmm_mle))
    println("GMM objective value: ", Optim.minimum(result_gmm_mle))


# Question 2(c): GMM with Random Starting Values

    Random.seed!(1234)
    alpha_random = rand(length(Optim.minimizer(result_mle)))

    result_gmm_random = optimize(
    alpha -> mlogit_gmm_overid(alpha, X_occ, y_occ),
    alpha_random,
    LBFGS(),
    Optim.Options(g_tol=1e-5, iterations=100_000)
    )

    println("Question 2(c): GMM with Random Starting Values")
    println("GMM estimates: ", Optim.minimizer(result_gmm_random))
    println("Optimization converged: ", Optim.converged(result_gmm_random))
    println("GMM objective value: ", Optim.minimum(result_gmm_random))
    println("Maximum difference from Q2(b): ",
    maximum(abs.(Optim.minimizer(result_gmm_random) .-
                     Optim.minimizer(result_gmm_mle))))

# Comparison of Q2(b) and Q2(c)
# Both optimizations converged, but the parameter estimates differ.
# The GMM objective value is lower with MLE starting values
# (1699.8202) than with random starting values (1701.4134).
# This suggests sensitivity to starting values and possible local minima.
# The results do not establish global concavity of the objective function.


# Question 3: Simulate Multinomial Logit Data

    Random.seed!(1234)

    Y_sim, X_sim = sim_logit(5_000, 4)

    println("Question 3: Multinomial Logit Simulation")
    println("X dimensions: ", size(X_sim))
    println("Y length: ", length(Y_sim))
    println("Choice categories: ", sort(unique(Y_sim)))
    println("Choice frequencies: ", [mean(Y_sim .== j) for j in 1:4])
    
# Question 3: Recover True Parameters via MLE

    alpha_sim_start = zeros(9)

    result_sim_mle = optimize(
        alpha -> mlogit_mle(alpha, X_sim, Y_sim),
        alpha_sim_start,
        LBFGS(),
        Optim.Options(g_tol=1e-5, iterations=100_000)
    )

    beta_true = [
        1.0  -2.0   0.0
        -1.0   0.5  -0.5
        0.5   0.3   2.0
    ]

    alpha_true = vec(beta_true)

    println("Question 3: Parameter Recovery")
    println("True parameters: ", alpha_true)
    println("Estimated parameters: ", Optim.minimizer(result_sim_mle))
    println("Optimization converged: ", Optim.converged(result_sim_mle))
    println("Maximum absolute difference: ",
    maximum(abs.(Optim.minimizer(result_sim_mle) .- alpha_true)))

# Question 3(f): Gumbel Simulation

    Y_gumbel, X_gumbel = sim_logit_w_gumbel(5_000, 4)

    println("Question 3(f): Gumbel Simulation")
    println("X dimensions: ", size(X_gumbel))
    println("Y length: ", length(Y_gumbel))
    println("Choice categories: ", sort(unique(Y_gumbel)))
    println("Choice frequencies: ",
        [mean(Y_gumbel .== j) for j in 1:4])

# Question 5: Multinomial Logit via SMM

D = 100

result_smm = optimize(
    alpha -> mlogit_smm_overid(alpha, X_occ, y_occ, D),
    Optim.minimizer(result_mle),
    NelderMead(),
    Optim.Options(iterations=1000)
)

println("Question 5: Multinomial Logit via SMM")
println("Number of observations: ", length(y_occ))
println("Number of simulation draws: ", D)
println("SMM estimates: ", Optim.minimizer(result_smm))
println("SMM objective value: ", Optim.minimum(result_smm))
println("Optimization converged: ", Optim.converged(result_smm))

println("Maximum absolute difference (SMM vs MLE): ",
        maximum(abs.(Optim.minimizer(result_smm) .-
                     Optim.minimizer(result_mle))))

    return nothing
end

@time main()



