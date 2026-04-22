using BayesianGammaARD
using Distributions
using DataFrames
using CSV
using Statistics
using Plots

# Vraies valeurs des paramètres
αtrue = 0.8
βtrue = 1
θtrue = 0.7
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
df25_rho0_2  = df[1:25,:]
df50_rho0_2  = df[1:50,:]
df75_rho0_2  = df[1:75,:]
df100_rho0_2 = df[1:100,:]
df125_rho0_2 = df[1:125,:]
df150_rho0_2 = df[1:150,:]
df175_rho0_2 = df[1:175,:]
df200_rho0_2 = df[1:200,:]

ρlow = Vector{Float64}(undef,9)
EstMLE_rho0_2 = Vector{Vector{Float64}}(undef,9)

res_NI_all_rho0_2 = Vector{DataFrame}(undef,9)
res_GP_all_rho0_2 = Vector{DataFrame}(undef,9)
res_BP_all_rho0_2 = Vector{DataFrame}(undef,9)


# Plot de la dégradation simulée
plot(df.tinsp, df.deg, label="Simulated degradation", xlabel="Time", ylabel="Degradation level", title="Simulated degradation process (ρ=0.2)")
hline!([3], label="L")
hline!([6], label="M")





for i in 1:length(ρlow)
    dfs = [df10_rho0_2, df25_rho0_2, df50_rho0_2, df75_rho0_2, df100_rho0_2,
           df125_rho0_2, df150_rho0_2, df175_rho0_2, df200_rho0_2]
    mydf = BayesianGammaARD.predf(gp, dfs[i])
    ρlow[i] = BayesianGammaARD.lowerboundrho(mydf)
    x0 = [1.0, 1.2, 1.0, (1+ρlow[i])/2]

    EstMLE_rho0_2[i] = MLE(gp, mydf, x0, [1e-2,0.1,0.01,ρlow[i]+0.01], [Inf, Inf, Inf, 1])
    # ... (bayesian estimation) ...
    priors_NI = [
        NonInformative(:α),
        NonInformative(:β),
        NonInformative(:θ),
        (1-ρlow[i])*Uniform() + ρlow[i]
    ]
    res_NI = algoMCMC(gp, mydf, priors_NI, 10000, 1.0, 0.5, 0.2)
    res_NI_all_rho0_2[i] = res_NI[1001:end,:]

    # "good priors" informatifs
    priormeanθ = θtrue
    priorvarθ = 0.2
    a = 2 + priormeanθ^2 / priorvarθ
    b = (a-1) * priormeanθ
    dθ = Informative(:θ, a, b, 1)

    # La moyenne a priori de β est 1
    # La variance est 1/c
    c = 2
    dβ = Informative(:β, c, c, 2)

    w = 1
    priormeanEw = θtrue * αtrue * w^βtrue
    priorvarEw = 0.2
    f = priorvarEw / priormeanEw
    e = priormeanEw / f
    dα = Informative(:α, e, f, 2)

    priormeanρ = 0.2
    priorvarρ = 0.02
    if priormeanρ <= ρlow[i]
        priormeanρ = (ρlow[i] + 1) / 2
        println("Warning: priormeanρ ($priormeanρ) is not greater than ρlow ($(ρlow[i])). $priormeanρ changed.")
    end
    g = (ρlow[i] - priormeanρ) * (ρlow[i] - ρlow[i]*priormeanρ - priormeanρ + priormeanρ^2 + priorvarρ) / (priorvarρ*(1-ρlow[i]))
    h = -(1-priormeanρ) * (ρlow[i] - ρlow[i]*priormeanρ - priormeanρ + priormeanρ^2 + priorvarρ) / (priorvarρ*(1-ρlow[i]))

    if g > 0 && h > 0
        dρ = (1-ρlow[i])*Informative(:ρ,g,h,1)+ρlow[i]
    else
        println("Warning: Invalid parameters for Beta distribution.")
        priormeanρ = ρlow[i] + 0.1
        println("New priors for ρ: mean = $priormeanρ, variance = $priorvarρ")
        g = (ρlow[i]-priormeanρ)*(ρlow[i]-ρlow[i]*priormeanρ-priormeanρ+priormeanρ^2+priorvarρ)/(priorvarρ*(1-ρlow[i]))
        h=-(1-priormeanρ)*(ρlow[i]-ρlow[i]*priormeanρ-priormeanρ+priormeanρ^2+priorvarρ)/(priorvarρ*(1-ρlow[i]))    
    end


    dρ = (1-ρlow[i]) * Informative(:ρ, g, h, 1) + ρlow[i]

    priors_GP = [dα, dβ, dθ, dρ]
    res_GP = algoMCMC(gp, mydf, priors_GP, 10000, 1.0, 0.5, 0.2)
    res_GP_all_rho0_2[i] = res_GP[1001:end,:]

    # Bad priors pour ρ

    badpriormeanρ = 0.7
    badpriorvarρ = 0.1
    if badpriormeanρ <= ρlow[i]
        badpriormeanρ = (ρlow[i] + 1) / 2
        println("Warning: badpriormeanρ ($badpriormeanρ) is not greater than ρlow ($(ρlow[i])). $badpriormeanρ changed.")
    end
    g_bad = (ρlow[i] - badpriormeanρ) * (ρlow[i] - ρlow[i]*badpriormeanρ - badpriormeanρ + badpriormeanρ^2 + badpriorvarρ) / (badpriorvarρ*(1-ρlow[i]))
    h_bad = -(1-badpriormeanρ) * (ρlow[i] - ρlow[i]*badpriormeanρ - badpriormeanρ + badpriormeanρ^2 + badpriorvarρ) / (badpriorvarρ*(1-ρlow[i]))

    if g_bad > 0 && h_bad > 0
        dρ_bad = (1-ρlow[i])*Informative(:ρ,g_bad,h_bad,1)+ρlow[i]
    else
        println("Warning: Invalid parameters for Beta distribution.")
        badvarpriorρ = 0.02  # Variance réduite
        g_bad = (ρlow[i]-badpriormeanρ)*(ρlow[i]-ρlow[i]*badpriormeanρ-badpriormeanρ+badpriormeanρ^2+badvarpriorρ)/(badvarpriorρ*(1-ρlow[i]))
        h_bad=-(1-badpriormeanρ)*(ρlow[i]-ρlow[i]*badpriormeanρ-badpriormeanρ+badpriormeanρ^2+badvarpriorρ)/(badvarpriorρ*(1-ρlow[i]))    
    end

    dρ_bad = (1-ρlow[i]) * Informative(:ρ, g_bad, h_bad, 1) + ρlow[i]

    priors_BP = [dα, dβ, dθ, dρ_bad]
    res_BP = algoMCMC(gp, mydf, priors_BP, 10000, 1.0, 0.5, 0.2)
    res_BP_all_rho0_2[i] = res_BP[1001:end,:]
end

# Sauvegarde (fichiers + noms de dataframes) avec indication rho 0.2
CSV.write("examples/Results/Homogeneous/sim_data_full_$rho_label.csv", df200_rho0_2)

n_values = [10,25,50,75,100,125,150,175,200]
for i in 1:length(res_NI_all_rho0_2)
    CSV.write("examples/Results/Homogeneous/bayes_posterior_non_informative_n$(n_values[i])_$rho_label.csv", res_NI_all_rho0_2[i])
    CSV.write("examples/Results/Homogeneous/bayes_posterior_good_priors_n$(n_values[i])_$rho_label.csv", res_GP_all_rho0_2[i])
    CSV.write("examples/Results/Homogeneous/bayes_posterior_bad_priors_n$(n_values[i])_$rho_label.csv", res_BP_all_rho0_2[i])
end

mle_df_rho_0_2 = DataFrame(n = Int[], α = Float64[], β = Float64[], θ = Float64[], ρ = Float64[])
for i in 1:length(EstMLE_rho0_2)
    push!(mle_df_rho_0_2, (n_values[i], EstMLE_rho0_2[i]...))
end
CSV.write("examples/Results/Homogeneous/mle_results_$rho_label.csv", mle_df_rho_0_2)
