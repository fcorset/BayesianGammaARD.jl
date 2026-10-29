using BayesianGammaARD
using DataFrames
using Random
using SpecialFunctions
using Statistics
using Test

@testset "rand_gig" begin
    Random.seed!(1234)

    for p in (0.5, 0.0, -0.5)
        a, b = 2.0, 3.0
        sample = [BayesianGammaARD.rand_gig(p, a, b) for _ in 1:100_000]
        z = sqrt(a * b)
        expected_mean = sqrt(b / a) * besselk(p + 1, z) / besselk(p, z)
        expected_variance = b / a * (
            besselk(p + 2, z) / besselk(p, z) -
            (besselk(p + 1, z) / besselk(p, z))^2
        )

        @test all(x -> isfinite(x) && x > 0, sample)
        @test mean(sample) ≈ expected_mean atol=0.03
        @test var(sample) ≈ expected_variance atol=0.06
    end

    @test_throws ArgumentError BayesianGammaARD.rand_gig(0.5, 0.0, 3.0)
    @test_throws ArgumentError BayesianGammaARD.rand_gig(0.5, 2.0, Inf)
end

@testset "MCMC diagnostics" begin
    samples = DataFrame(
        α = [1.0, 2.0, 3.0, 4.0, 5.0],
        β = [1.0, 1.0, 2.0, 2.0, 3.0],
        θ = [2.0, 2.0, 2.0, 2.0, 2.0],
        ρ = [0.1, 0.2, 0.3, 0.4, 0.5],
    )
    diagnostics = mcmc_diagnostics(samples; max_lag=2)

    @test diagnostics.acceptance_rates.α == 1.0
    @test diagnostics.acceptance_rates.β == 0.5
    @test ismissing(diagnostics.acceptance_rates.θ)
    @test diagnostics.acceptance_rates.ρ == 1.0
    @test diagnostics.autocorrelation.lag == 0:2
    @test diagnostics.autocorrelation.α[1] == 1.0
    @test 1 <= diagnostics.ess[:α] <= nrow(samples)
    @test ismissing(diagnostics.ess[:θ])
    @test all(ismissing, diagnostics.autocorrelation.θ)
end
