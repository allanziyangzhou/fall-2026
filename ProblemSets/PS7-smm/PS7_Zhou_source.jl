
# PS7 - Question 1
# Load data for OLS estimation

function load_data(url)

    df = CSV.read(HTTP.get(url).body, DataFrame)

    X = [ones(size(df, 1), 1) df.age df.race .== 1 df.collgrad .== 1]
    y = Float64.(df.married .== 1)

    return df, X, y
end


# PS7 - Question 1
# OLS estimation via GMM

function ols_gmm(beta, X, y)

    # Compute the moment vector
    g = y - X * beta

    # Identity weighting matrix: W = I
    obj = dot(g, g)

    return obj
end


# PS7 - Question 2
# Prepare occupation data for multinomial logit

function prepare_occupation_data(df)

    df = dropmissing(copy(df), :occupation)

    # Combine occupation categories 8 through 13 into category 7
    for j in 8:13
        df[df.occupation .== j, :occupation] .= 7
    end

    # Create explanatory variables
    df.white = df.race .== 1

    X = [ones(size(df, 1), 1) df.age df.white df.collgrad]
    y = Int.(df.occupation)

    return df, X, y
end


# PS7 - Question 2(a)
# Multinomial Logit via Maximum Likelihood

function mlogit_mle(alpha, X, y)

    N, K = size(X)
    J = length(alpha) ÷ K + 1

    # Normalize the last alternative to zero
    alpha_mat = hcat(reshape(alpha, K, J - 1), zeros(K))

    # Compute utility indices
    V = X * alpha_mat

    # Compute log probabilities using log-sum-exp
    row_max = maximum(V, dims=2)
    log_denom = row_max .+ log.(sum(exp.(V .- row_max), dims=2))
    logP = V .- log_denom

    # Compute negative log-likelihood
    loglike = -sum(logP[i, y[i]] for i in 1:N)

    return loglike
end


# PS7 - Question 2(b)
# Overidentified GMM for Multinomial Logit

function mlogit_gmm_overid(alpha, X, y)

    N, K = size(X)
    J = length(alpha) ÷ K + 1

    # Create observed choice indicators
    bigY = zeros(N, J)
    for j in 1:J
        bigY[:, j] .= (y .== j)
    end

    # Normalize the last alternative
    alpha_mat = hcat(reshape(alpha, K, J - 1), zeros(K))

    # Compute predicted choice probabilities
    V = X * alpha_mat
    row_max = maximum(V, dims=2)
    expV = exp.(V .- row_max)
    P = expV ./ sum(expV, dims=2)

    # Stack the moment differences into an N*J vector
    g = vec(bigY .- P)

    # Identity weighting matrix
    obj = dot(g, g)

    return obj
end


# PS7 - Question 3
# Simulate data from a multinomial logit model

function sim_logit(N=100_000, J=4)

    # Question 3(a): Generate covariates
    X = hcat(
        ones(N),
        randn(N),
        2 .+ 2 .* randn(N)
    )

    # Question 3(b): Set true parameter values
    if J == 4
        beta = hcat(
            [1.0, -1.0, 0.5],
            [-2.0, 0.5, 0.3],
            [0.0, -0.5, 2.0],
            zeros(3)
        )
    else
        beta = hcat(
            2 .* rand(3, J - 1) .- 1,
            zeros(3)
        )
    end

    # Question 3(c): Compute choice probabilities
    V = X * beta

    row_max = maximum(V, dims=2)
    expV = exp.(V .- row_max)
    P = expV ./ sum(expV, dims=2)

    # Question 3(d): Draw uniform random shocks
    epsilon = rand(N)

    
    # Question 3(e): Generate choices using inverse CDF

    Y = zeros(Int, N)

    for j in 1:J
        Y .+= vec(epsilon .< sum(P[:, j:J], dims=2))
    end

    return Y, X

end

# Question 3(f)
# Simulate multinomial logit choices using Gumbel shocks

function sim_logit_w_gumbel(N=100_000, J=4)

    # Generate covariates
    X = hcat(
        ones(N),
        randn(N),
        2 .+ 2 .* randn(N)
    )

    # Set true parameter values
    if J == 4
        beta = hcat(
            [1.0, -1.0, 0.5],
            [-2.0, 0.5, 0.3],
            [0.0, -0.5, 2.0],
            zeros(3)
        )
    else
        beta = hcat(
            2 .* rand(3, J - 1) .- 1,
            zeros(3)
        )
    end

    # Draw independent Gumbel preference shocks
    epsilon = rand(Gumbel(0, 1), N, J)

    # Choose the alternative with the highest utility
    V = X * beta .+ epsilon
    Y = [argmax(row) for row in eachrow(V)]

    return Y, X
end


# PS7 - Question 5
# Multinomial Logit via Simulated Method of Moments

function mlogit_smm_overid(alpha, X, y, D)

    N, K = size(X)
    J = length(alpha) ÷ K + 1

    # Create observed choice indicators
    bigY = zeros(N, J)

    # Fill the observed choice indicator matrix
    for j in 1:J
        bigY[:, j] .= (y .== j)
    end

    # Initialize simulated choice frequencies
    bigY_sim = zeros(N, J)

    # Reshape parameters and normalize the base alternative
    alpha_mat = hcat(
        reshape(alpha, K, J - 1),
        zeros(K)
    )
    # Simulate choice outcomes using Gumbel shocks
    Random.seed!(1234)

    for d in 1:D

        epsilon = rand(Gumbel(0, 1), N, J)

        # Compute utilities and simulated choices
        V = X * alpha_mat .+ epsilon
        y_sim = [argmax(row) for row in eachrow(V)]

        # Accumulate simulated choice frequencies
        for j in 1:J
            bigY_sim[:, j] .+= (y_sim .== j) ./ D
        end

    end

        # Compute moment differences
    g = vec(bigY .- bigY_sim)

    # Compute the SMM objective function
    obj = dot(g, g)

    return obj

end
