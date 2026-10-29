module BayesianGammaARD
using Distributions
using DataFrames
using SpecialFunctions
using Optim
using Random
using Statistics
using CSV

export GammaProcess, MaintenanceModel, MLE, loglikelihood, NonInformative, pdf, Informative, params, algoMCMC, mcmc_diagnostics, predf, lowerboundrho, deltaeta, logcondpostdistbeta, logcondpostdistalpha, logcondpostdistrho, rand_gig, mcmc_diagnostics
# Write your package code here.
include("maintenancemodels.jl")
include("gammaprocesses.jl")
include("bayesian.jl")
end
