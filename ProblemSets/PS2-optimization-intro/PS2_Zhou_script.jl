using Optim
using Random
using DataFrames
using CSV
using HTTP
using GLM
using LinearAlgebra
using FreqTables

include("PS2_Zhou_source.jl")
function main()
# Question 1

startval = rand(1)

result_q1 = solve_q1(startval)

println("Question 1")
println("Starting value: ", startval)
println("argmax is ", Optim.minimizer(result_q1)[1])
println("maximum is ", f(Optim.minimizer(result_q1)))

# Question 2

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

X = [ones(size(df,1),1) df.age df.race .== 1 df.collgrad .== 1]

y = df.married .== 1

beta_hat_ols = optimize(
    b -> ols(b, X, y),
    rand(size(X,2)),
    LBFGS(),
    Optim.Options(
        g_tol = 1e-6,
        iterations = 100_000,
        show_trace = true
    )
)

println("Question 2")
println("OLS estimates from Optim:")
println(beta_hat_ols.minimizer)

bols = inv(X' * X) * X' * y

df.white = df.race .== 1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)

println("OLS estimates from matrix formula:")
println(bols)

println("OLS estimates from GLM:")
println(coef(bols_lm))

# Question 3

beta_hat_logit = optimize(
    a -> -logit(a, X, y),
    rand(size(X,2)),
    LBFGS(),
    Optim.Options(
        g_tol = 1e-6,
        iterations = 100_000,
        show_trace = true
    )
)

println("Question 3")
println("Logit estimates from Optim:")
println(beta_hat_logit.minimizer)

# Question 4

df.white = df.race .== 1

logit_glm = glm(
    @formula(married ~ age + white + collgrad),
    df,
    Binomial(),
    LogitLink()
)

println("Question 4")
println("Logit estimates from GLM:")
println(coef(logit_glm))

# Question 5

freqtable(df, :occupation)

df = dropmissing(df, :occupation)

df[df.occupation .== 8,  :occupation] .= 7
df[df.occupation .== 9,  :occupation] .= 7
df[df.occupation .== 10, :occupation] .= 7
df[df.occupation .== 11, :occupation] .= 7
df[df.occupation .== 12, :occupation] .= 7
df[df.occupation .== 13, :occupation] .= 7

freqtable(df, :occupation)

X = [ones(size(df,1),1) df.age df.race .== 1 df.collgrad .== 1]

y = df.occupation

alpha0 = zeros(size(X, 2) * 6)

result_q5 = optimize(
    a -> -mlogit(a, X, y),
    alpha0,
    LBFGS(),
    Optim.Options(
        g_tol = 1e-5,
        iterations = 100_000,
        show_trace = true
    )
)

println("Question 5")
println("Multinomial logit estimates from Optim:")
coef_q5 = reshape(result_q5.minimizer, size(X, 2), 6)

println(coef_q5)

end

main()