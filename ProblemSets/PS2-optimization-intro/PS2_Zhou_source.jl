# Question 1

function f(x)
    return -x[1]^4 - 10x[1]^3 - 2x[1]^2 - 3x[1] - 2
end

function minusf(x)
    return x[1]^4 + 10x[1]^3 + 2x[1]^2 + 3x[1] + 2
end

function solve_q1(startval)
    return optimize(minusf, startval, LBFGS())
end

# Question 2

function ols(beta, X, y)
    ssr = (y .- X * beta)' * (y .- X * beta)
    return ssr
end

# Question 3

function logit(alpha, X, d)

    xb = X * alpha

    loglike = sum(d .* xb .- log.(1 .+ exp.(xb)))

    return loglike
end

# Question 5

function mlogit(alpha, X, d)

    K = size(X, 2)
    A = reshape(alpha, K, 6)

    V = X * A

    loglike = 0.0

    for i in 1:size(X, 1)

        denom = 1 + sum(exp.(V[i, :]))

        if d[i] == 7
            loglike += -log(denom)
        else
            j = Int(d[i])
            loglike += V[i, j] - log(denom)
        end

    end

    return loglike
end