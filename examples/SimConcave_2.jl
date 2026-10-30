# In this script, we simulate a concave degradation process and low efficiency of Preventive Maintenance 
# and perform Bayesian estimation with different priors (good, moderately and strongly misspecified priors).

using BayesianGammaARD
using Distributions
using DataFrames
using CSV
using Statistics
using Plots


# Vraies valeurs des paramètres
αtrue = 2.0
βtrue = 0.6
θtrue = 2.0
ρtrue = 0.2
rho_label = "rho0.2"

# Définition du modèle
mm = MaintenanceModel(ρ=ρtrue)
gp = GammaProcess(α=αtrue,β=βtrue,θ=θtrue,mm=mm)

# ... (simulation) ...

HT = 200 # Fenêtre d'observation [0,T]


# Simulation du processus de dégradation
y, df = rand(gp,tinsp=1:HT,HT=HT)


# On génère 9 jeux de données (n croissant) avec nom de variable explicite
df10_rho0_2  = df[1:10,:]
df20_rho0_2  = df[1:20,:]
df30_rho0_2  = df[1:30,:]
df50_rho0_2  = df[1:50,:]
df75_rho0_2  = df[1:75,:]
df100_rho0_2 = df[1:100,:]
df125_rho0_2 = df[1:125,:]
df150_rho0_2 = df[1:150,:]
df175_rho0_2 = df[1:175,:]
df200_rho0_2 = df[1:200,:]

# Cas où l'on veut charger les données depuis des fichiers CSV
df10_rho0_2 = CSV.read("examples/Results/Concave/sim_data_full_$rho_label.csv", DataFrame)[1:10,:]
df20_rho0_2 = CSV.read("examples/Results/Concave/sim_data_full_$rho_label.csv", DataFrame)[1:20,:]
df30_rho0_2 = CSV.read("examples/Results/Concave/sim_data_full_$rho_label.csv", DataFrame)[1:30,:]
df50_rho0_2 = CSV.read("examples/Results/Concave/sim_data_full_$rho_label.csv", DataFrame)[1:50,:]
df75_rho0_2 = CSV.read("examples/Results/Concave/sim_data_full_$rho_label.csv", DataFrame)[1:75,:]
df100_rho0_2 = CSV.read("examples/Results/Concave/sim_data_full_$rho_label.csv", DataFrame)[1:100,:]
df125_rho0_2 = CSV.read("examples/Results/Concave/sim_data_full_$rho_label.csv", DataFrame)[1:125,:]
df150_rho0_2 = CSV.read("examples/Results/Concave/sim_data_full_$rho_label.csv", DataFrame)[1:150,:]
df175_rho0_2 = CSV.read("examples/Results/Concave/sim_data_full_$rho_label.csv", DataFrame)[1:175,:]
df200_rho0_2 = CSV.read("examples/Results/Concave/sim_data_full_$rho_label.csv", DataFrame)[1:200,:]


ρlow = Vector{Float64}(undef,10)
EstMLE_rho0_2 = Vector{Vector{Float64}}(undef,10)

# res_NI_all_rho0_2 = Vector{DataFrame}(undef,9)
res_GP_all_rho0_2 = [Vector{DataFrame}(undef,4) for _ in 1:10] # Good Priors
res_BP_all_rho0_2 = [Vector{DataFrame}(undef,4) for _ in 1:10] # moderately misspecified priors
res_VBP_all_rho0_2 = [Vector{DataFrame}(undef,4) for _ in 1:10] # strongly misspecified priors


for i in 1:length(ρlow)
    dfs = [df10_rho0_2, df20_rho0_2, df30_rho0_2, df50_rho0_2, df75_rho0_2,
           df100_rho0_2, df125_rho0_2, df150_rho0_2, df175_rho0_2, df200_rho0_2]
    mydf = BayesianGammaARD.predf(gp, dfs[i])
    ρlow[i] = BayesianGammaARD.lowerboundrho(mydf)
    println("Lower bound for ρ (n=$(nrow(dfs[i]))): $(ρlow[i])")
    x0 = [1.0, 1.5, 1.0, (1+ρlow[i])/2]

    EstMLE_rho0_2[i] = MLE(gp, mydf, x0, [1e-2,0.1,0.01,ρlow[i]+0.01], [Inf, Inf, Inf, 1])
    # ... (bayesian estimation) ...
    #priors_NI = [
    #    NonInformative(:α),
    #    NonInformative(:β),
    #    NonInformative(:θ),
    #    (1-ρlow[i])*Uniform() + ρlow[i]
    #]
    #res_NI = algoMCMC(gp, mydf, priors_NI, 10000, 1.0, 0.5, 0.2)
    #res_NI_all_rho0_2[i] = res_NI[1001:end,:]

    # Faire 3 cas :
    # - correctly centered informative priors, 
    # - moderately misspecified priors, 
    # - strongly misspecified priors
    
    # Pour chaque simulation, donner le nombre de maintenances correctives, 
    # le nombre de maintenances préventives, 

    # For MCMC Algorithm, we need to output:
    # - the values of the estimated parameters by MLE
    # - values of parameters estimated by Bayes with informative priors
    # - acceptance rates for each parameter, autocorrelation, multiple-chain diagnostics

    # Recommandation: The authors should report the proposal scales, tuning procedure, acceptance rates, effective sample
    # sizes, trace plots, and a multiple-chain diagnostic such as R. They should explain how proposals
    # outside the parameter space are treated. If truncated normal proposals are used, the Hastings ratio
    # must account for the proposal-density normalization. If ordinary normal proposals are used, the
    # proposal density must be normalized.

    ############################
    # "good priors"
    ############################
    priormeanθ = θtrue
    priorvarθ = 0.5
    a_g = 2 + priormeanθ^2 / priorvarθ
    b_g = (a_g-1) * priormeanθ
    dθ_g = Informative(:θ, a_g, b_g, 1)

    priormeanβ = βtrue
    priorvarβ = 0.1
    κ_g = priormeanβ * (1-priormeanβ) / priorvarβ - 1
    c_g = priormeanβ * κ_g
    d_g = (1-priormeanβ) * κ_g
    println("Parameters for β: c = $c_g, d = $d_g")

    dβ_g = Informative(:β, c_g, d_g, 1) # le troisième paramètre est égal à 1 pour loi beta

    # Give a time horizon w for the prior of α, where E_w=αθw^β is the mean of degradation at time w wihtout maintenance. 
    # The prior for α is a gamma distribution with mean Ew/(θ*w^β) and variance 0.2/(θ*w^β)^2, where Ew is the expected degradation at time w.
    # w should be depend of L, the parameter of the maintenance model, but for simplicity, we can take w=1.
    # We choose w where the probability of corrective maintenance is close to 0.5, i.e. P(CM at time w) = 0.5. 
    
    w = 1.0
    priormeanEw = θtrue * αtrue * w^βtrue
    priorvarEw = 0.5
    f_g = priorvarEw / priormeanEw
    e_g = priormeanEw / f_g
    dα_g = Informative(:α, e_g, f_g, 1)

    priormeanρ = 0.2
    priorvarρ = 0.05
    # The following condition is obsolte since we keep the indicator in likelihood
    # if priormeanρ <= ρlow[i]
    #     priormeanρ = (ρlow[i] + 1) / 2
    #     println("Warning: priormeanρ ($priormeanρ) is not greater than ρlow ($(ρlow[i])). $priormeanρ changed.")
    # end
    kappa_g = priormeanρ * (1-priormeanρ) / priorvarρ - 1
    g_g = priormeanρ * kappa_g
    h_g = (1-priormeanρ) * kappa_g
    
    dρ_g = Informative(:ρ,g_g,h_g,1)
    priors_GP = ContinuousUnivariateDistribution[dα_g, dβ_g, dθ_g, dρ_g]
    #priors_GP = [dα_g, dβ_g, dθ_g, dρ_g]
    #########################################
    ## Moderately misspecified priors
    #########################################
    badpriormeanθ = θtrue + 0.5
    badpriorvarθ = 0.8
    a_b = 2 + badpriormeanθ^2 / badpriorvarθ
    b_b = (a_b-1) * badpriormeanθ
    dθ_b = Informative(:θ, a_b, b_b, 1)

    badpriormeanβ = βtrue + 0.15
    badpriorvarβ = 0.1
    κ_b = badpriormeanβ * (1-badpriormeanβ) / badpriorvarβ - 1
    c_b = badpriormeanβ * κ_b
    d_b = (1-badpriormeanβ) * κ_b
    dβ_b = Informative(:β, c_b, d_b, 1)

    badpriormeanEw = θtrue * αtrue * w^βtrue + 0.5
    badpriorvarEw = 1
    f_b = badpriorvarEw / badpriormeanEw
    e_b = badpriormeanEw / f_b
    dα_b = Informative(:α, e_b, f_b, 1)

    badpriormeanρ = 0.4
    badpriorvarρ = 0.1

    kappa_b = badpriormeanρ * (1-badpriormeanρ) / badpriorvarρ - 1
    g_b = badpriormeanρ * kappa_b
    h_b = (1-badpriormeanρ) * kappa_b

    dρ_b = Informative(:ρ,g_b,h_b,1)
    
    priors_BP = [dα_g, dβ_g, dθ_g, dρ_b]

    #########################################
    ## Strongly misspecified priors
    #########################################
    verybadpriormeanθ = θtrue + 1.0
    verybadpriorvarθ = 0.5
    a_vb = 2 + verybadpriormeanθ^2 / verybadpriorvarθ
    b_vb = (a_vb-1) * verybadpriormeanθ
    dθ_vb = Informative(:θ, a_vb, b_vb, 1)

    verybadpriormeanβ = βtrue + 0.3
    verybadpriorvarβ = 0.07
    κ_vb = verybadpriormeanβ * (1-verybadpriormeanβ) / verybadpriorvarβ - 1
    c_vb = verybadpriormeanβ * κ_vb
    d_vb = (1-verybadpriormeanβ) * κ_vb
    dβ_vb = Informative(:β, c_vb, d_vb, 1)

    verybadpriormeanEw = θtrue * αtrue * w^βtrue + 2.0
    verybadpriorvarEw = 1
    f_vb = verybadpriorvarEw / verybadpriormeanEw
    e_vb = verybadpriormeanEw / f_vb
    dα_vb = Informative(:α, e_vb, f_vb, 1)

    verybadpriormeanρ = 0.7
    verybadpriorvarρ = 0.2 
    g_vb = verybadpriormeanρ * (1-verybadpriormeanρ) / verybadpriorvarρ - 1
    h_vb = (1-verybadpriormeanρ) * g_vb
    dρ_vb = Informative(:ρ,g_vb,h_vb,1) 


    priors_VBP = [dα_vb, dβ_vb, dθ_vb, dρ_vb]

    # ALgo MCMC
    K = 10000

    mle_start = EstMLE_rho0_2[i]
    
    initial_points = [
        [mle_start[1], mle_start[2], mle_start[3], clamp(mle_start[4], ρlow[i] + 1e-6, 1 - 1e-6)],
        [0.8 * mle_start[1], clamp(mle_start[2] - 0.05, 1e-6, 1 - 1e-6),
         1.1 * mle_start[3], clamp(mle_start[4] - 0.05, ρlow[i] + 1e-6, 1 - 1e-6)],
        [1.2 * mle_start[1], clamp(mle_start[2] + 0.05, 1e-6, 1 - 1e-6),
         0.9 * mle_start[3], clamp(mle_start[4] + 0.05, ρlow[i] + 1e-6, 1 - 1e-6)],
        [1.5 * mle_start[1], clamp(mle_start[2] - 0.1, 1e-6, 1 - 1e-6),
         1.2 * mle_start[3], clamp(mle_start[4] + 0.1, ρlow[i] + 1e-6, 1 - 1e-6)]
    ]

    for chain in 1:4
        println("Chaîne numéro $chain")
        res_GP = algoMCMC(gp, mydf, priors_GP, K, 1.0, 0.5, 0.2; w=w, x0=initial_points[chain])
        res_GP_all_rho0_2[i][chain] = res_GP[1001:end,:]

        res_BP = algoMCMC(gp, mydf, priors_BP, K, 1.0, 0.5, 0.2; w=w, x0=initial_points[chain])
        res_BP_all_rho0_2[i][chain] = res_BP[1001:end,:]

        res_VBP = algoMCMC(gp, mydf, priors_VBP, K, 1.0, 0.5, 0.2; w=w, x0=initial_points[chain])
        res_VBP_all_rho0_2[i][chain] = res_VBP[1001:end,:]
    end
end

# Sauvegarde (fichiers + noms de dataframes) avec indication rho 0.2
CSV.write("examples/Results/Concave/sim_data_full_$rho_label.csv", df200_rho0_2)

n_values = [10,20,30,50,75,100,125,150,175,200]
for i in 1:length(res_GP_all_rho0_2)
    for chain in 1:4
        CSV.write("examples/Results/Concave/bayes_posterior_verybad_priors_n$(n_values[i])_chain$(chain)_$rho_label.csv", res_VBP_all_rho0_2[i][chain])
        CSV.write("examples/Results/Concave/bayes_posterior_good_priors_n$(n_values[i])_chain$(chain)_$rho_label.csv", res_GP_all_rho0_2[i][chain])
        CSV.write("examples/Results/Concave/bayes_posterior_bad_priors_n$(n_values[i])_chain$(chain)_$rho_label.csv", res_BP_all_rho0_2[i][chain])
    end
end

mle_df_rho_0_2 = DataFrame(k = Int[], n = Int[], α = Float64[], β = Float64[], θ = Float64[], ρ = Float64[])
for i in 1:length(EstMLE_rho0_2)
    push!(mle_df_rho_0_2, (k, n_values[i], EstMLE_rho0_2[i]...))
end
mle_df_rho_0_2[!, :ρlow] = ρlow
CSV.write("examples/Results/Concave/mle_results_$rho_label.csv", mle_df_rho_0_2)
