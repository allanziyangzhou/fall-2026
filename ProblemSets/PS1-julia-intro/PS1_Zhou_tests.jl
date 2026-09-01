using Test

include("PS1_Zhou_source.jl")
@testset "matrixops tests" begin

    A = [1.0 2.0; 3.0 4.0]
    B = [5.0 6.0; 7.0 8.0]

    result1, result2, result3 = matrixops(A, B)

    @test result1 == A .* B
    @test result2 == A' * B
    @test result3 == sum(A + B)

    C = [1.0 2.0 3.0; 4.0 5.0 6.0]

@test_throws ErrorException matrixops(A, C)

end

@testset "q1 tests" begin

    A, B, C, D = q1()

@test size(A) == (10, 7)
@test size(B) == (10, 7)
@test size(C) == (5, 7)
@test size(D) == (10, 7)

end

@testset "q2 tests" begin

    A, B, C, D = q1()

    @test q2(A, B, C) === nothing

end

@testset "q3 tests" begin

    @test q3() === nothing

end

@testset "q4 tests" begin

    @test q4() === nothing

end