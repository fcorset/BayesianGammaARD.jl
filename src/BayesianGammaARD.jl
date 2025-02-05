module BayesianGammaARD
using Distributions
using DataFrames
using SpecialFunctions
using Optim
using Statistics
using CSV


export GammaProcess, MaintenanceModel, MLE, loglikelihood, NonInformative, pdf, Informative
# Write your package code here.
include("maintenancemodels.jl")
include("gammaprocesses.jl")
include("bayesian.jl")
end
