
# PS6 - Question 1
# Load bus data and reshape to long panel format

function load_and_reshape_data(url::String)

    # Read the original wide-format data
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Extract state variables for Question 3
    Xstate = Matrix(df[:, [Symbol("Xst", t) for t in 1:20]])
    Zstate = Vector(df[:, :Zst])
    Branded = Vector(df[:, :Branded])

    # Add bus ID
    df.bus_id = 1:nrow(df)

    # Reshape outcome and odometer variables
    df_long = stack(
        df,
        [Symbol("Y", t) for t in 1:20],
        variable_name = :period,
        value_name = :Y
    )

    df_long.time = parse.(Int, replace.(string.(df_long.period), "Y" => ""))

    odo_long = stack(
        df,
        [Symbol("Odo", t) for t in 1:20],
        variable_name = :period,
        value_name = :Odometer
    )

    odo_long.time = parse.(Int, replace.(string.(odo_long.period), "Odo" => ""))

    # Match odometer observations by bus and time
    df_long = leftjoin(
        df_long[:, [:bus_id, :time, :Y, :RouteUsage, :Branded]],
        odo_long[:, [:bus_id, :time, :Odometer]],
        on = [:bus_id, :time]
    )

    sort!(df_long, [:bus_id, :time])
    # Ensure Odometer is treated as continuous in GLM
    df_long.Odometer = Float64.(df_long.Odometer)

    return df_long, Xstate, Zstate, Branded
end


# PS6 - Question 2
# Estimate a fully interacted flexible logit model

function estimate_flexible_logit(df::DataFrame)

    # Create polynomial terms within the formula
    formula = @formula(
    Y ~ Odometer * abs2(Odometer) *
        RouteUsage * abs2(RouteUsage) *
        Branded * time * abs2(time)
)

    flex_logit = glm(formula, df, Binomial(), LogitLink())

    return flex_logit
end


# PS6 - Question 3(b)
# Construct the state space for CCP prediction

function construct_state_space(xbin::Int, zbin::Int,
                               xval::Vector, zval::Vector)

    state_df = DataFrame(
        Odometer = kron(ones(zbin), xval),
        RouteUsage = kron(zval, ones(xbin)),
        Branded = zeros(Int, xbin * zbin),
        time = zeros(Int, xbin * zbin)
    )

    return state_df
end

# PS6 - Question 3(b)
# Compute future values using conditional choice probabilities

function compute_future_values(state_df::DataFrame,
                               flex_logit,
                               xtran::Matrix,
                               xbin::Int,
                               T::Int,
                               β::Float64)

    # Initialize future value array
    FV = zeros(size(xtran, 1), 2, T + 1)

    # Compute CCP-based future values
    for t in 2:T
        for b in 0:1

            # Update the state variables
            state_df.time .= t
            state_df.Branded .= b

            # Probability of continuing to run the bus (Y = 1)
            p1 = predict(flex_logit, state_df)

            # Probability of replacing the engine (Y = 0)
            p0 = 1.0 .- p1

            # Avoid taking log(0) due to numerical rounding
            p0 = clamp.(p0, eps(Float64), 1.0)

            # Store discounted future value terms
            FV[:, b+1, t] .= -β .* log.(p0)

        end
    end

    return FV
end


# PS6 - Question 3(b)
# Compute future value differences for each observation

function compute_fvt1(df_long::DataFrame,
                      FV::Array{Float64, 3},
                      xtran::Matrix,
                      Xstate::Matrix,
                      Zstate::Vector,
                      xbin::Int,
                      B::Vector)

    nbus, T = size(Xstate)
    FVT1 = zeros(nbus, T)

    for i in 1:nbus
        for t in 1:T

            # Identify the current state and replacement state
            row0 = 1 + (Zstate[i] - 1) * xbin
            row1 = Xstate[i, t] + (Zstate[i] - 1) * xbin

            # Difference in transition probabilities
            transition_diff = xtran[row1, :] - xtran[row0, :]

            # Expected future value difference
            future_values = FV[row0:(row0 + xbin - 1), B[i] + 1, t + 1]

            FVT1[i, t] = dot(transition_diff, future_values)

        end
    end

    # Convert from bus-by-time matrix to long format
    fvt1 = vec(permutedims(FVT1))

    return fvt1
end


# PS6 - Question 3(c)
# Estimate structural parameters using CCPs

function estimate_structural_params(df_long::DataFrame,
                                    fvt1::Vector{Float64})

    # Create a copy to preserve the original data
    df = copy(df_long)

    # Add the future value difference as an offset
    df.fv = fvt1

    # Estimate the structural logit model
    structural_logit = glm(
        @formula(Y ~ Odometer + Branded),
        df,
        Binomial(),
        LogitLink(),
        offset = df.fv
    )

    return structural_logit
end


