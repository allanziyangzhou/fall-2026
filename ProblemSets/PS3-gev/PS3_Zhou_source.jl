# ============================================================
# PS3_Zhou_source.jl
# ECON 6343 - Problem Set 3
# ============================================================


# ============================================================
# Question 1: Multinomial Logit with Alternative-Specific Z
# ============================================================


# ------------------------------------------------------------
# Unpack the parameter vector
# ------------------------------------------------------------
function unpack_mnl_params(theta, K, J)

    # First K*(J-1) parameters are the beta coefficients.
    # Alternative J is normalized so that beta_J = 0.
    beta = reshape(theta[1:K*(J-1)], K, J-1)

    # The final parameter is gamma,
    # the coefficient on the alternative-specific variable Z.
    gamma = theta[end]

    return beta, gamma
end


# ------------------------------------------------------------
# Compute multinomial logit choice probabilities
# ------------------------------------------------------------
function mnl_probabilities(theta, X, Z)

    n, K = size(X)
    J = size(Z, 2)

    beta, gamma = unpack_mnl_params(theta, K, J)

    # Utility matrix:
    # rows = individuals
    # columns = alternatives
    V = zeros(n, J)

    # Utilities for alternatives 1,...,J-1
    for j in 1:(J-1)

        V[:, j] =
            X * beta[:, j] .+
            gamma .* (Z[:, j] .- Z[:, J])

    end

    # Alternative J is the normalized alternative.
    # beta_J = 0 and Z_J - Z_J = 0,
    # so V[:, J] remains equal to zero.

    # Numerically stable softmax
    maxV = maximum(V, dims=2)

    expV = exp.(V .- maxV)

    P = expV ./ sum(expV, dims=2)

    return P
end


# ------------------------------------------------------------
# Negative log-likelihood
# ------------------------------------------------------------
function mnl_negloglik(theta, X, Z, y)

    P = mnl_probabilities(theta, X, Z)

    n = length(y)

    loglik = 0.0

    for i in 1:n

        chosen_alternative = Int(y[i])

        loglik += log(P[i, chosen_alternative])

    end

    # Optim minimizes objective functions,
    # so return the negative log-likelihood.
    return -loglik
end


# ------------------------------------------------------------
# Estimate the multinomial logit model
# ------------------------------------------------------------
function estimate_mnl(X, Z, y)

    K = size(X, 2)
    J = size(Z, 2)

    # There are:
    # K coefficients for each of alternatives 1,...,J-1
    # plus one gamma coefficient.
    theta0 = zeros(K * (J - 1) + 1)

    # Objective function passed to Optim
    objective(theta) = mnl_negloglik(theta, X, Z, y)

    # Maximum likelihood estimation:
    # minimize the negative log-likelihood.
    result = Optim.optimize(
        objective,
        theta0,
        Optim.BFGS(),
        Optim.Options(iterations = 5000)
    )

    # Estimated parameter vector
    theta_hat = Optim.minimizer(result)

    # Separate beta and gamma estimates
    beta_hat, gamma_hat =
        unpack_mnl_params(theta_hat, K, J)

    return result, theta_hat, beta_hat, gamma_hat
end

# ============================================================
# Question 3: Nested Logit
# ============================================================


# ------------------------------------------------------------
# Unpack nested-logit parameters
# ------------------------------------------------------------
function unpack_nested_params(theta, K)

    beta_WC = theta[1:K]
    beta_BC = theta[(K + 1):(2 * K)]

    lambda_WC = theta[2 * K + 1]
    lambda_BC = theta[2 * K + 2]

    gamma = theta[2 * K + 3]

    return beta_WC, beta_BC, lambda_WC, lambda_BC, gamma
end


# ------------------------------------------------------------
# Row-wise log-sum-exp
# ------------------------------------------------------------
function row_logsumexp(M)

    m = maximum(M, dims = 2)

    return vec(
        m .+ log.(sum(exp.(M .- m), dims = 2))
    )
end


# ------------------------------------------------------------
# Nested-logit log probabilities
# ------------------------------------------------------------
function nested_logit_logprobabilities(theta, X, Z)

    n, K = size(X)
    J = size(Z, 2)

    beta_WC,
    beta_BC,
    lambda_WC,
    lambda_BC,
    gamma =
        unpack_nested_params(theta, K)


    # --------------------------------------------------------
    # Nest-level X beta
    # --------------------------------------------------------

    xb_WC = X * beta_WC
    xb_BC = X * beta_BC


    # --------------------------------------------------------
    # Difference Z relative to alternative 8
    # --------------------------------------------------------

    Zdiff = Z[:, 1:7] .- reshape(Z[:, J], n, 1)


    # --------------------------------------------------------
    # Utilities inside WC nest: alternatives 1, 2, 3
    # --------------------------------------------------------

    utility_WC =
        (
            reshape(xb_WC, n, 1) .+
            gamma .* Zdiff[:, 1:3]
        ) ./ lambda_WC


    # --------------------------------------------------------
    # Utilities inside BC nest: alternatives 4, 5, 6, 7
    # --------------------------------------------------------

    utility_BC =
        (
            reshape(xb_BC, n, 1) .+
            gamma .* Zdiff[:, 4:7]
        ) ./ lambda_BC


    # --------------------------------------------------------
    # Log of within-nest sums
    # --------------------------------------------------------

    logsum_WC = row_logsumexp(utility_WC)
    logsum_BC = row_logsumexp(utility_BC)


    # --------------------------------------------------------
    # Overall denominator
    #
    # 1
    # + [sum_WC exp(.)]^lambda_WC
    # + [sum_BC exp(.)]^lambda_BC
    # --------------------------------------------------------

    denominator_terms = hcat(
        zeros(n),
        lambda_WC .* logsum_WC,
        lambda_BC .* logsum_BC
    )

    log_denom = row_logsumexp(denominator_terms)


    # --------------------------------------------------------
    # Log choice probabilities
    # --------------------------------------------------------

    logP = zeros(n, J)


    # White collar: alternatives 1-3

    logP[:, 1:3] .=
        utility_WC .+
        reshape(
            (lambda_WC - 1.0) .* logsum_WC .- log_denom,
            n,
            1
        )


    # Blue collar: alternatives 4-7

    logP[:, 4:7] .=
        utility_BC .+
        reshape(
            (lambda_BC - 1.0) .* logsum_BC .- log_denom,
            n,
            1
        )


    # Alternative 8: Other
    # beta_Other = 0

    logP[:, J] .= -log_denom


    return logP
end


# ------------------------------------------------------------
# Ordinary probabilities
# ------------------------------------------------------------
function nested_logit_probabilities(theta, X, Z)

    return exp.(nested_logit_logprobabilities(theta, X, Z))
end


# ------------------------------------------------------------
# Negative log-likelihood
# ------------------------------------------------------------
function nested_logit_negloglik(theta, X, Z, y)

    logP = nested_logit_logprobabilities(theta, X, Z)

    loglik = 0.0

    @inbounds for i in eachindex(y)

        loglik += logP[i, Int(y[i])]

    end

    return -loglik
end


# ------------------------------------------------------------
# Estimate nested-logit model
# ------------------------------------------------------------
function estimate_nested_logit(X, Z, y)

    K = size(X, 2)

    number_of_parameters = 2 * K + 3


    # --------------------------------------------------------
    # Initial values
    # --------------------------------------------------------

    theta0 = zeros(number_of_parameters)

    # Start lambda values away from the boundaries
    theta0[2 * K + 1] = 0.8
    theta0[2 * K + 2] = 0.8


    # --------------------------------------------------------
    # Parameter bounds
    # --------------------------------------------------------

    lower = fill(-Inf, number_of_parameters)
    upper = fill( Inf, number_of_parameters)

    # 0 < lambda <= 1
    lower[2 * K + 1] = 1.0e-4
    lower[2 * K + 2] = 1.0e-4

    upper[2 * K + 1] = 1.0
    upper[2 * K + 2] = 1.0


    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    objective(theta) =
        nested_logit_negloglik(theta, X, Z, y)


    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    result = Optim.optimize(
        objective,
        lower,
        upper,
        theta0,
        Optim.Fminbox(Optim.LBFGS()),
        Optim.Options(
            iterations = 1000,
            g_tol = 1.0e-6,
            f_tol = 1.0e-10,
            show_trace = true,
            show_every = 50
        )
    )


    # --------------------------------------------------------
    # Extract estimates
    # --------------------------------------------------------

    theta_hat = Optim.minimizer(result)

    beta_WC_hat,
    beta_BC_hat,
    lambda_WC_hat,
    lambda_BC_hat,
    gamma_hat =
        unpack_nested_params(theta_hat, K)


    return (
        result,
        theta_hat,
        beta_WC_hat,
        beta_BC_hat,
        lambda_WC_hat,
        lambda_BC_hat,
        gamma_hat
    )
end