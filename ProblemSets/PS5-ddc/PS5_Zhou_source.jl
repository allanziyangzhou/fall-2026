
# PS5 - Question 1
# Load and reshape data from wide to long panel format

function load_static_data()
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Create bus id variable
    df = @transform(df, :bus_id = 1:size(df,1))

    # Reshape decision variable (Y1-Y20)
    dfy = @select(df, :bus_id, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10,
                  :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20,
                  :RouteUsage, :Branded)

    dfy_long = DataFrames.stack(dfy, Not([:bus_id, :RouteUsage, :Branded]))
    rename!(dfy_long, :value => :Y)
    dfy_long = @transform(dfy_long, :time = kron(collect(1:20), ones(size(df,1))))
    select!(dfy_long, Not(:variable))

    # Reshape odometer variable (Odo1-Odo20)
    dfx = @select(df, :bus_id, :Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7,
                  :Odo8, :Odo9, :Odo10, :Odo11, :Odo12, :Odo13, :Odo14, :Odo15,
                  :Odo16, :Odo17, :Odo18, :Odo19, :Odo20)

    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    rename!(dfx_long, :value => :Odometer)
    dfx_long = @transform(dfx_long, :time = kron(collect(1:20), ones(size(df,1))))
    select!(dfx_long, Not(:variable))

    # Join reshaped dataframes
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id, :time])
    sort!(df_long, [:bus_id, :time])

    return df_long
end


# PS5 - Question 2
# Estimate the static binary logit model

function estimate_static_model(df_long)
    model = glm(@formula(Y ~ Odometer + Branded),
                df_long, Binomial(), LogitLink())

    return model
end


# PS5 - Question 3(a)
# Load dynamic model data in wide format

function load_dynamic_data()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS5-ddc/busdata.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    Y = Matrix(df[:, r"^Y\d+$"])
    X = Matrix(df[:, r"^Odo\d+$"])
    Xstate = Matrix(df[:, r"^Xst\d+$"])
    Zstate = Vector(df[:, :Zst])
    B = Vector(df[:, :Branded])

    N, T = size(Y)

    zval, zbin, xval, xbin, xtran = create_grids()

    return (
        Y = Y,
        X = X,
        Xstate = Xstate,
        Zstate = Zstate,
        B = B,
        N = N,
        T = T,
xval = xval,
xbin = xbin,
zbin = zbin,
xtran = xtran,
β = 0.9
    )
end


# PS5 - Question 3(c)
# Compute future values using backward recursion

@views @inbounds function compute_future_value!(FV, θ, d)

    fill!(FV, 0.0)

    for t in d.T:-1:1
        for b in 0:1
            for z in 1:d.zbin
                for x in 1:d.xbin

                    row = x + (z - 1) * d.xbin
                    row0 = 1 + (z - 1) * d.xbin
                    rows = row0:(row0 + d.xbin - 1)

                    # Continue running the bus
                    v1 = θ[1] + θ[2] * d.xval[x] + θ[3] * b +
                         dot(d.xtran[row, :], FV[rows, b+1, t+1])

                    # Replace the engine
                    v0 = dot(d.xtran[row0, :], FV[rows, b+1, t+1])

                    # Discounted future value, stable log-sum-exp
                    m = max(v0, v1)
                    FV[row, b+1, t] = d.β *
                        (m + log(exp(v0-m) + exp(v1-m)))

                end
            end
        end
    end

    return FV
end


# PS5 - Question 3(d)
# Compute log likelihood using observed states

@views @inbounds function compute_log_likelihood(θ, d, FV)

    loglike = 0.0

    for i in 1:d.N

        # Row index after engine replacement
        row0 = 1 + (d.Zstate[i] - 1) * d.xbin
        rows = row0:(row0 + d.xbin - 1)

        for t in 1:d.T

            # Row index for continuing with current engine
            row1 = d.Xstate[i, t] + (d.Zstate[i] - 1) * d.xbin

            # Current flow utility difference
            flow_diff = θ[1] + θ[2] * d.X[i, t] +
                        θ[3] * d.B[i]

            # Difference in expected future values
            ev_diff = dot(
                d.xtran[row1, :] .- d.xtran[row0, :],
                FV[rows, d.B[i] + 1, t + 1]
            )

            # Conditional value difference
            v_diff = flow_diff + ev_diff

            # Log likelihood contribution (stable logit form)
            loglike += d.Y[i, t] * v_diff -
                       (max(0.0, v_diff) +
                        log1p(exp(-abs(v_diff))))

        end
    end

    return loglike
end


# PS5 - Question 3(e)
# Negative log likelihood for optimization

@views @inbounds function negative_log_likelihood(θ, d)

    FV = zeros(d.zbin * d.xbin, 2, d.T + 1)

    compute_future_value!(FV, θ, d)

    loglike = compute_log_likelihood(θ, d, FV)

    return -loglike
end
