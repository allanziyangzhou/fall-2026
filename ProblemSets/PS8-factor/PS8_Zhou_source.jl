
# PS8 - Question 1
# Load data and estimate the baseline wage regression

function load_data(url::String)
    return CSV.read(HTTP.get(url).body, DataFrame)
end

function estimate_base_regression(df::DataFrame)
    return lm(
        @formula(logwage ~ black + hispanic + female +
                 schoolt + gradHS + grad4yr),
        df
    )
end


# PS8 - Question 2
# Correlations among six ASVAB scores

function compute_asvab_correlations(df::DataFrame)

    asvab_cols = [:asvabAR, :asvabCS, :asvabMK,
                  :asvabNO, :asvabPC, :asvabWK]

    asvab_matrix = Matrix{Float64}(df[:, asvab_cols])

    return cor(asvab_matrix)
end

# PS8 - Question 3
# Wage regression with all six ASVAB scores

function estimate_full_regression(df::DataFrame)
    return lm(
        @formula(logwage ~ black + hispanic + female +
                 schoolt + gradHS + grad4yr +
                 asvabAR + asvabCS + asvabMK +
                 asvabNO + asvabPC + asvabWK),
        df
    )
end

# PS8 - Question 4
# Wage regression using the first principal component of ASVAB scores

function estimate_pca_regression(df::DataFrame)

    asvab_cols = [:asvabAR, :asvabCS, :asvabMK,
                  :asvabNO, :asvabPC, :asvabWK]

    # PCA requires variables x observations (6 x N)
    asvab_matrix = permutedims(Matrix{Float64}(df[:, asvab_cols]))

    # Extract the first principal component
    pca_model = fit(PCA, asvab_matrix; maxoutdim=1)
    pca_scores = vec(MultivariateStats.transform(pca_model, asvab_matrix))

    # Add principal component scores to a copy of the data
    df_pca = copy(df)
    df_pca.asvabPCA = pca_scores

    # Estimate the wage regression with the first principal component
    return lm(
        @formula(logwage ~ black + hispanic + female +
                 schoolt + gradHS + grad4yr + asvabPCA),
        df_pca
    )
end

# PS8 - Question 5
# Wage regression using the first factor of ASVAB scores

function estimate_factor_regression(df::DataFrame)

    asvab_cols = [:asvabAR, :asvabCS, :asvabMK,
                  :asvabNO, :asvabPC, :asvabWK]

    # FactorAnalysis requires variables x observations (6 x N)
    asvab_matrix = permutedims(Matrix{Float64}(df[:, asvab_cols]))

    # Extract one common factor
    factor_model = fit(FactorAnalysis, asvab_matrix; maxoutdim=1)
    factor_scores = vec(MultivariateStats.transform(factor_model, asvab_matrix))

    # Add factor scores without modifying the original data
    df_factor = copy(df)
    df_factor.asvabFactor = factor_scores

    # Estimate the wage regression with the extracted factor
    return lm(
        @formula(logwage ~ black + hispanic + female +
                 schoolt + gradHS + grad4yr + asvabFactor),
        df_factor
    )
end

# PS8 - Question 6
# Prepare data matrices for the full factor model

function prepare_factor_matrices(df::DataFrame)

    # Wage equation: intercept + six covariates (N x 7)
    X = hcat(
        ones(nrow(df)),
        Matrix{Float64}(df[:, [:black, :hispanic, :female,
                               :schoolt, :gradHS, :grad4yr]])
    )

    # Log wage (N observations)
    y = Float64.(df.logwage)

    # Measurement equations: intercept + three covariates (N x 4)
    Xfac = hcat(
        ones(nrow(df)),
        Matrix{Float64}(df[:, [:black, :hispanic, :female]])
    )

    # Six ASVAB measurements (N x 6)
    asvabs = Matrix{Float64}(
        df[:, [:asvabAR, :asvabCS, :asvabMK,
               :asvabNO, :asvabPC, :asvabWK]]
    )

    return X, y, Xfac, asvabs
end

# Question 6: Unpack factor model parameters

function unpack_factor_parameters(theta::AbstractVector)

    alpha = reshape(theta[1:24], 4, 6)
    beta = theta[25:31]
    gamma = theta[32:37]
    delta = theta[38]
    sigma = exp.(theta[39:45])

    return alpha, beta, gamma, delta, sigma
end

# Question 6: Negative log-likelihood for the full factor model

function factor_negloglik(theta, X, y, Xfac, asvabs, nodes, weights)

    alpha, beta, gamma, delta, sigma =
        unpack_factor_parameters(theta)

    N = length(y)
    Q = length(nodes)

    # Predicted values excluding the latent factor
    mu_asvab = Xfac * alpha
    mu_wage = X * beta

    # Log-likelihood conditional on each quadrature node
    logL = zeros(eltype(theta), N, Q)
    log2pi = log(2 * pi)

    for j in 1:6
        residual = (asvabs[:, j] .- mu_asvab[:, j]) .-
                   gamma[j] .* reshape(nodes, 1, :)

        logL .-= 0.5 .* (residual ./ sigma[j]).^2 .+
                 log(sigma[j]) .+ 0.5 * log2pi
    end

    wage_residual = (y .- mu_wage) .-
                    delta .* reshape(nodes, 1, :)

    logL .-= 0.5 .* (wage_residual ./ sigma[7]).^2 .+
             log(sigma[7]) .+ 0.5 * log2pi

    # Integrate out the unobserved standard normal factor
    log_weights = log.(weights) .+ logpdf.(Normal(), nodes)
    logL .+= reshape(log_weights, 1, :)

    # Numerically stable log-sum-exp
    row_max = maximum(logL, dims=2)
    integrated_logL = row_max .+
                      log.(sum(exp.(logL .- row_max), dims=2))

    return -sum(integrated_logL)
end

# Question 6: Initialize MLE parameters

function initialize_factor_parameters(X, y, Xfac, asvabs)

    alpha0 = Xfac \ asvabs
    beta0 = X \ y

    residual_asvab = asvabs - Xfac * alpha0
    residual_wage = y - X * beta0

    gamma0 = fill(0.2, 6)
    delta0 = 0.1

    sigma0 = vcat(
        vec(std(residual_asvab, dims=1)),
        std(residual_wage)
    )

    theta0 = vcat(
        vec(alpha0),
        beta0,
        gamma0,
        delta0,
        log.(sigma0)
    )

    return theta0
end
