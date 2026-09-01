# ECON 6343: Econometrics III
# Problem Set 1
# Name: Ziyang Zhou

using JLD
using Random
using Distributions
using CSV
using DataFrames
using Statistics
using FreqTables

function q1()

    # Question 1(a)
    Random.seed!(1234)

    # i. A: 10 x 7 matrix, Uniform[-5, 10]
    A = rand(Uniform(-5, 10), 10, 7)

    # ii. B: 10 x 7 matrix, Normal(mean = -2, sd = 15)
    B = rand(Normal(-2, 15), 10, 7)

    # iii. C: first 5 rows and first 5 columns of A,
    # plus first 5 rows and last 2 columns of B
    C = hcat(A[1:5, 1:5], B[1:5, 6:7])

    # iv. D: keep A[i,j] if A[i,j] <= 0, otherwise 0
    D = ifelse.(A .<= 0, A, 0.0)

    # Question 1(b)
    num_elements_A = length(A)
    println("Number of elements in A: ", num_elements_A)
    
    # Question 1(c)
    num_unique_D = length(unique(D))
    println("Number of unique elements in D: ", num_unique_D)

    # Question 1(d)
    E = reshape(B, 70, 1)   

    # Easier way:
    # E = vec(B)
    println("Size of E: ", size(E))

    # Question 1(e)
    F = cat(A, B; dims=3)
    println("Size of F: ", size(F))

    # Question 1(f)
    F = permutedims(F, (3, 1, 2))
    println("New size of F: ", size(F))

    # Question 1(g)
    G = kron(B, C)
    println("Size of G: ", size(G))

    # Try C ⊗ F
    try
    CF = kron(C, F)
    println("Size of C kron F: ", size(CF))
    catch err
    println("C kron F gives an error: ", err)
    end

    # Question 1(h)
    save(
    "matrixpractice.jld",
    "A", A,
    "B", B,
    "C", C,
    "D", D,
    "E", E,
    "F", F,
    "G", G
    )

    println("Saved matrixpractice.jld")

    # Question 1(i)
    save(
    "firstmatrix.jld",
    "A", A,
    "B", B,
    "C", C,
    "D", D
    )

    println("Saved firstmatrix.jld")

    # Question 1(j)
    C_df = DataFrame(C, :auto)
    CSV.write("Cmatrix.csv", C_df)

    println("Saved Cmatrix.csv")

    # Question 1(k)
    D_df = DataFrame(D, :auto)
    CSV.write("Dmatrix.dat", D_df; delim='\t')

    println("Saved Dmatrix.dat")
    
    # Q1(l)
    return A, B, C, D
    end

function q2(A, B, C)

    # Q2(a): using loops
    AB = similar(A)

    for i in axes(A, 1)
        for j in axes(A, 2)
            AB[i, j] = A[i, j] * B[i, j]
        end
    end

    # Q2(a): without loops or comprehensions
    AB2 = A .* B


    # Q2(b): using a loop
    Cprime = Float64[]

    for x in C
        if -5 <= x <= 5
            push!(Cprime, x)
        end
    end

    # Q2(b): without a loop
    Cprime2 = C[(-5 .<= C) .& (C .<= 5)]

# Q2(c)
N = 15169
K = 6
T = 5

X = zeros(N, K, T)

# Columns 1, 5, and 6 are stationary over time
col1 = ones(N)
col5 = rand(Binomial(20, 0.6), N)
col6 = rand(Binomial(20, 0.5), N)

for t in 1:T

    X[:, 1, t] = col1

    p = 0.75 * (6 - t) / 5
    X[:, 2, t] = rand(Bernoulli(p), N)

    if t == 1
        X[:, 3, t] .= 15
    else
        X[:, 3, t] = rand(Normal(15 + t - 1, 5 * (t - 1)), N)
    end

    X[:, 4, t] = rand(Normal(pi * (6 - t) / 3, 1 / exp(1)), N)

    X[:, 5, t] = col5
    X[:, 6, t] = col6

end

# Q2(d)
β = [
    k == 1 ? 1 + 0.25 * (t - 1) :
    k == 2 ? log(t) :
    k == 3 ? -sqrt(t) :
    k == 4 ? exp(t) - exp(t + 1) :
    k == 5 ? t :
             t / 3
    for k in 1:K, t in 1:T
]

# Q2(e)
Y = hcat([
    X[:, :, t] * β[:, t] + rand(Normal(0, 0.36), N)
    for t in 1:T
]...)

# Q2(f)
return nothing
end

function q3()
# Q3(a)
nlsw88 = CSV.read(
    "ProblemSets/PS1-julia-intro/nlsw88.csv",
    DataFrame;
    normalizenames = true
)

CSV.write(
    "ProblemSets/PS1-julia-intro/nlsw88_processed.csv",
    nlsw88
)

# Q3(b)
pct_never_married = mean(nlsw88.never_married) * 100
pct_collgrad = mean(nlsw88.collgrad) * 100

println("Never married: ", pct_never_married, "%")
println("College graduates: ", pct_collgrad, "%")

# Q3(c)
race_freq = freqtable(nlsw88.race)
race_pct = race_freq ./ sum(race_freq) .* 100

println(race_pct)

# Q3(d)
summarystat = describe(
    nlsw88,
    :mean,
    :median,
    :std,
    :min,
    :max,
    :nunique,
    :nmissing
)

# Q3(e)
industry_occupation = freqtable(nlsw88.industry, nlsw88.occupation)

println(industry_occupation)

# Q3(f)
wage_data = select(nlsw88, [:industry, :occupation, :wage])

grouped_wage = groupby(wage_data, [:industry, :occupation])

mean_wage = combine(grouped_wage, :wage => mean => :mean_wage)

# Q3(g)
return nothing
end

A, B, C, D = q1()
q2(A, B, C)
q3()

# Q4(b), Q4(c), Q4(e)
"""
    matrixops(A, B)

Takes matrices A and B as inputs and returns:
1. the element-by-element product of A and B,
2. the matrix product A' * B,
3. the sum of all elements of A + B.
"""
function matrixops(A, B)

    if size(A) != size(B)
        error("inputs must have the same size.")
    end

    result1 = A .* B
    result2 = A' * B
    result3 = sum(A + B)

    return result1, result2, result3
end

function q4()
# Q4(a)
firstmatrix = load("ProblemSets/PS1-julia-intro/firstmatrix.jld")

# Q4(d)
result1, result2, result3 =
    matrixops(firstmatrix["A"], firstmatrix["B"])
    
# Q4(f)
# matrixops(firstmatrix["C"], firstmatrix["D"])
# This gives an error because C and D do not have the same size.

# Q4(g)
nlsw88_processed = CSV.read(
    "ProblemSets/PS1-julia-intro/nlsw88_processed.csv",
    DataFrame
)

ttl_exp_array = convert(Array, nlsw88_processed.ttl_exp)
wage_array = convert(Array, nlsw88_processed.wage)

matrixops(ttl_exp_array, wage_array)

# Q4(h)
return nothing
end