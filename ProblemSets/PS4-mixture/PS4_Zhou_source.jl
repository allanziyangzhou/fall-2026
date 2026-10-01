# ============================================================
# PS4_Zhou_source.jl
# ECON 6343 - Problem Set 4
# ============================================================


# ============================================================
# Question 1:
# Multinomial Logit with Alternative-Specific Covariates Z
# ============================================================


# ------------------------------------------------------------
# Unpack parameter vector
# ------------------------------------------------------------
function unpack_mlogit_params(theta, K, J)

    # First K*(J-1) parameters are coefficients on X.
    # Alternative J is normalized so that beta_J = 0.
    alpha = reshape(
        theta[1:(K * (J - 1))],
        K,
        J - 1
    )

    # Last parameter is the coefficient on Z.
    gamma = theta[end]

    return alpha, gamma
end


# ------------------------------------------------------------
# Compute multinomial logit choice probabilities
# ------------------------------------------------------------
function mlogit_probabilities(theta, X, Z)

    N = size(X, 1)
    K = size(X, 2)
    J = size(Z, 2)

    alpha, gamma =
        unpack_mlogit_params(theta, K, J)


    # --------------------------------------------------------
    # Add zero coefficients for normalized alternative J
    # --------------------------------------------------------

    T = promote_type(eltype(X), eltype(theta))

    bigAlpha = [
        alpha zeros(T, K)
    ]


    # --------------------------------------------------------
    # Utility for each alternative
    #
    # V_ij =
    # X_i * beta_j + gamma * (Z_ij - Z_iJ)
    # --------------------------------------------------------

    V = zeros(T, N, J)

    for j in 1:J

        V[:, j] =
            X * bigAlpha[:, j] .+
            gamma .* (Z[:, j] .- Z[:, J])

    end


    # --------------------------------------------------------
    # Numerically stable multinomial-logit probabilities
    # --------------------------------------------------------

    maxV = maximum(V, dims = 2)

    expV = exp.(V .- maxV)

    P = expV ./ sum(expV, dims = 2)


    return P
end


# ------------------------------------------------------------
# Negative log-likelihood
# ------------------------------------------------------------
function mlogit_with_Z(theta, X, Z, y)

    P = mlogit_probabilities(theta, X, Z)

    N = length(y)

    loglike = zero(eltype(P))


    # --------------------------------------------------------
    # Add log probability of the alternative actually chosen
    # --------------------------------------------------------

    for i in 1:N

        chosen_alternative = Int(y[i])

        loglike += log(
            P[i, chosen_alternative]
        )

    end


    # Optim minimizes, so return negative log-likelihood.
    return -loglike
end


# ------------------------------------------------------------
# Estimate multinomial logit
# using automatic differentiation
# ------------------------------------------------------------
function estimate_mlogit_with_Z(
    X,
    Z,
    y;
    theta0 = nothing
)

    K = size(X, 2)
    J = size(Z, 2)

    number_of_parameters =
        K * (J - 1) + 1


    # --------------------------------------------------------
    # Starting values
    # --------------------------------------------------------

    if theta0 === nothing

        theta_start =
            zeros(number_of_parameters)

    else

        theta_start =
            copy(theta0)

    end


    # --------------------------------------------------------
    # Objective function
    # --------------------------------------------------------

    objective(theta) =
        mlogit_with_Z(
            theta,
            X,
            Z,
            y
        )


    # --------------------------------------------------------
    # Optimization with automatic differentiation
    # --------------------------------------------------------

    result = Optim.optimize(
        objective,
        theta_start,
        Optim.BFGS();
        autodiff = AutoForwardDiff()
    )


    # --------------------------------------------------------
    # Estimated parameters
    # --------------------------------------------------------

    theta_hat =
        Optim.minimizer(result)

    alpha_hat,
    gamma_hat =
        unpack_mlogit_params(
            theta_hat,
            K,
            J
        )


    # --------------------------------------------------------
    # Standard errors
    #
    # Variance-covariance matrix =
    # inverse Hessian of negative log-likelihood
    # --------------------------------------------------------

    H =
        ForwardDiff.hessian(
            objective,
            theta_hat
        )

    vcov =
        inv(H)

    se_theta =
        sqrt.(diag(vcov))


    # --------------------------------------------------------
    # Separate standard errors
    # --------------------------------------------------------

    alpha_se =
        reshape(
            se_theta[
                1:(K * (J - 1))
            ],
            K,
            J - 1
        )

    gamma_se =
        se_theta[end]


    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return (
        result = result,
        theta_hat = theta_hat,
        alpha_hat = alpha_hat,
        gamma_hat = gamma_hat,
        se_theta = se_theta,
        alpha_se = alpha_se,
        gamma_se = gamma_se,
        vcov = vcov
    )
end

# ============================================================
# Question 3: Quadrature and Monte Carlo Integration
# ============================================================


# ------------------------------------------------------------
# Question 3(a):
# Quadrature for a Normal(0,1)
# ------------------------------------------------------------
function quadrature_normal_check()

    d = Normal(0, 1)

    # 7 quadrature points over +/- 4 standard deviations
    nodes, weights = lgwt(7, -4, 4)

    # Integral of the density should be approximately 1
    density_integral =
        sum(weights .* pdf.(d, nodes))

    # Integral of x*f(x) should be approximately 0
    mean_integral =
        sum(weights .* nodes .* pdf.(d, nodes))

    return density_integral, mean_integral
end


# ------------------------------------------------------------
# Question 3(b):
# Quadrature approximation to variance of Normal(0,2)
# ------------------------------------------------------------
function quadrature_variance(n_points)

    # Normal(mean = 0, standard deviation = 2)
    d = Normal(0, 2)

    sigma = 2.0

    # +/- 5 sigma
    lower = -5 * sigma
    upper =  5 * sigma

    nodes, weights =
        lgwt(n_points, lower, upper)

    # Integral of x^2 f(x)
    variance_integral =
        sum(
            weights .*
            nodes.^2 .*
            pdf.(d, nodes)
        )

    return variance_integral
end


# ------------------------------------------------------------
# Question 3(c):
# Monte Carlo integration
# ------------------------------------------------------------
function monte_carlo_integrals(D; seed = 1234)

    Random.seed!(seed)

    d = Normal(0, 2)

    sigma = 2.0

    lower = -5 * sigma
    upper =  5 * sigma

    # Uniform draws over [lower, upper]
    Xdraws =
        lower .+
        (upper - lower) .* rand(D)

    mc_weight =
        (upper - lower) / D


    # Integral of x^2 f(x)
    second_moment =
        mc_weight *
        sum(
            Xdraws.^2 .*
            pdf.(d, Xdraws)
        )


    # Integral of x f(x)
    first_moment =
        mc_weight *
        sum(
            Xdraws .*
            pdf.(d, Xdraws)
        )


    # Integral of f(x)
    density_integral =
        mc_weight *
        sum(
            pdf.(d, Xdraws)
        )


    return (
        second_moment = second_moment,
        first_moment = first_moment,
        density_integral = density_integral
    )
end

# ============================================================
# Question 4: Mixed Logit with Quadrature
# DO NOT RUN THE ESTIMATION
# ============================================================


function mixed_logit_quad(theta, X, Z, y, nodes, weights)

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    K = size(X, 2)
    J = size(Z, 2)
    N = length(y)


    # --------------------------------------------------------
    # Parameters
    #
    # First K*(J-1) elements: alpha
    # Next element: mean of gamma distribution
    # Last element: standard deviation of gamma distribution
    # --------------------------------------------------------

    alpha =
        theta[1:(K * (J - 1))]

    mu_gamma =
        theta[end - 1]

    sigma_gamma =
        theta[end]


    # --------------------------------------------------------
    # Choice indicator matrix
    # --------------------------------------------------------

    bigY = zeros(N, J)

    for j in 1:J
        bigY[:, j] = y .== j
    end


    # --------------------------------------------------------
    # Normalize alternative J
    # --------------------------------------------------------

    T = promote_type(
        eltype(X),
        eltype(theta)
    )

    bigAlpha = [
        reshape(alpha, K, J - 1) zeros(T, K)
    ]


    # --------------------------------------------------------
    # Integrated probability for each observation
    # --------------------------------------------------------

    integrated_prob =
        zeros(T, N)


    # --------------------------------------------------------
    # Loop over quadrature nodes
    # --------------------------------------------------------

    for r in eachindex(nodes)

        gamma_r = nodes[r]


        # ----------------------------------------------------
        # Utility at quadrature node r
        # ----------------------------------------------------

        V =
            zeros(T, N, J)

        for j in 1:J

            V[:, j] =
                X * bigAlpha[:, j] .+
                gamma_r .* (
                    Z[:, j] .- Z[:, J]
                )

        end


        # ----------------------------------------------------
        # Conditional multinomial-logit probabilities
        # ----------------------------------------------------

        maxV =
            maximum(V, dims = 2)

        expV =
            exp.(V .- maxV)

        P =
            expV ./ sum(expV, dims = 2)


        # ----------------------------------------------------
        # Probability of the alternative actually chosen
        # ----------------------------------------------------

        chosen_prob =
            vec(
                sum(
                    bigY .* P,
                    dims = 2
                )
            )


        # ----------------------------------------------------
        # Normal density f(gamma_r)
        #
        # gamma ~ N(mu_gamma, sigma_gamma^2)
        # ----------------------------------------------------

        density_r =
            exp(
                -0.5 *
                (
                    (gamma_r - mu_gamma) /
                    sigma_gamma
                )^2
            ) /
            (
                sigma_gamma *
                sqrt(2 * pi)
            )


        # ----------------------------------------------------
        # Quadrature approximation
        #
        # Integral ≈ sum_r
        # weight_r * probability_r * density_r
        # ----------------------------------------------------

        integrated_prob .+=
            weights[r] .*
            chosen_prob .*
            density_r

    end


    # --------------------------------------------------------
    # Negative log-likelihood
    # --------------------------------------------------------

    loglike =
        -sum(
            log.(integrated_prob)
        )


    return loglike
end