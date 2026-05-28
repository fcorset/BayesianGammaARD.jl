using BayesianGammaARD
using Distributions
using DataFrames
using CSV
using Statistics
using Plots
using DrWatson

# Vraies valeurs des paramètres
αtrue = 0.8
βtrue = 1
θtrue = 1
ρtrue = 0.7
rho_label = "rho0.7"

# Définition du modèle
mm = MaintenanceModel(ρ=ρtrue)
gp = GammaProcess(α=αtrue,β=βtrue,θ=θtrue,mm=mm)

# ... (simulation) ...

HT = 200 # Fenêtre d'observation [0,T]

# Simulation du processus de dégradation
y, df = rand(gp,tinsp=1:HT,HT=HT)


# On génère 9 jeux de données (n croissant) avec nom de variable explicite
df10_rho0_7  = df[1:10,:]
df25_rho0_7  = df[1:25,:]
df50_rho0_7  = df[1:50,:]
df75_rho0_7  = df[1:75,:]
df100_rho0_7 = df[1:100,:]
df125_rho0_7 = df[1:125,:]
df150_rho0_7 = df[1:150,:]
df175_rho0_7 = df[1:175,:]
df200_rho0_7 = df[1:200,:]

# Cas où l'on veut charger les données depuis des fichiers CSV
df10_rho0_7 = CSV.read("examples/Results/Homogeneous/sim_data_full_$rho_label.csv", DataFrame)[1:10,:]
df25_rho0_7 = CSV.read("examples/Results/Homogeneous/sim_data_full_$rho_label.csv", DataFrame)[1:25,:]
df50_rho0_7 = CSV.read("examples/Results/Homogeneous/sim_data_full_$rho_label.csv", DataFrame)[1:50,:]
df75_rho0_7 = CSV.read("examples/Results/Homogeneous/sim_data_full_$rho_label.csv", DataFrame)[1:75,:]
df100_rho0_7 = CSV.read("examples/Results/Homogeneous/sim_data_full_$rho_label.csv", DataFrame)[1:100,:]
df125_rho0_7 = CSV.read("examples/Results/Homogeneous/sim_data_full_$rho_label.csv", DataFrame)[1:125,:]
df150_rho0_7 = CSV.read("examples/Results/Homogeneous/sim_data_full_$rho_label.csv", DataFrame)[1:150,:]
df175_rho0_7 = CSV.read("examples/Results/Homogeneous/sim_data_full_$rho_label.csv", DataFrame)[1:175,:]
df200_rho0_7 = CSV.read("examples/Results/Homogeneous/sim_data_full_$rho_label.csv", DataFrame)[1:200,:]



ρlow = Vector{Float64}(undef,9)
EstMLE_rho0_7 = Vector{Vector{Float64}}(undef,9)

res_NI_all_rho0_7 = Vector{DataFrame}(undef,9)
res_GP_all_rho0_7 = Vector{DataFrame}(undef,9)


# Plot de la dégradation simulée
plot(df200_rho0_7.tinsp, df200_rho0_7.deg, label="Simulated degradation", xlabel="Time", ylabel="Degradation level", title="Simulated degradation process (ρ=0.7)")
hline!([3], label="l_PM")
hline!([6], label="l_CM")




for i in 1:length(ρlow)
    dfs = [df10_rho0_7, df25_rho0_7, df50_rho0_7, df75_rho0_7, df100_rho0_7,
           df125_rho0_7, df150_rho0_7, df175_rho0_7, df200_rho0_7]
    mydf = BayesianGammaARD.predf(gp, dfs[i])
    ρlow[i] = BayesianGammaARD.lowerboundrho(mydf)
    x0 = [1.0, 1.2, 1.0, (1+ρlow[i])/2]

    EstMLE_rho0_7[i] = MLE(gp, mydf, x0, [1e-2,0.1,0.01,ρlow[i]+0.01], [Inf, Inf, Inf, 1])
    # ... (bayesian estimation) ...
    priors_NI = [
        NonInformative(:α),
        NonInformative(:β),
        NonInformative(:θ),
        (1-ρlow[i])*Uniform() + ρlow[i]
    ]
    res_NI = algoMCMC(gp, mydf, priors_NI, 10000, 1.0, 0.5, 0.2)
    res_NI_all_rho0_7[i] = res_NI[1001:end,:]

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

    priormeanρ = 0.7
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
    res_GP_all_rho0_7[i] = res_GP[1001:end,:]
end

# Sauvegarde (fichiers + noms de dataframes) avec indication rho 0.7
CSV.write("examples/Results/Homogeneous/sim_data_full_$rho_label.csv", df200_rho0_7)

n_values = [10,25,50,75,100,125,150,175,200]
for i in 1:length(res_NI_all_rho0_7)
    CSV.write("examples/Results/Homogeneous/bayes_posterior_non_informative_n$(n_values[i])_$rho_label.csv", res_NI_all_rho0_7[i])
    CSV.write("examples/Results/Homogeneous/bayes_posterior_good_priors_n$(n_values[i])_$rho_label.csv", res_GP_all_rho0_7[i])
end

mle_df_rho_0_7 = DataFrame(n = Int[], α = Float64[], β = Float64[], θ = Float64[], ρ = Float64[])
for i in 1:length(EstMLE_rho0_7)
    push!(mle_df_rho_0_7, (n_values[i], EstMLE_rho0_7[i]...))
end
mle_df_rho_0_7[!, :ρlow] = ρlow
CSV.write("examples/Results/Homogeneous/mle_results_$rho_label.csv", mle_df_rho_0_7)
