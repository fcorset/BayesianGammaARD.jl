# fichier pour analyser les données de la simulation et faire des graphiques
using Plots
using CSV
using DataFrames
using BayesianGammaARD
using DrWatson

# Cas Convex : ρ = 0.7
# Charger les données simulées en récupérant les valeurs des paramètres
# Vraies valeurs des paramètres
# αtrue = 0.8
# βtrue = 1.5
# θtrue = 1.0
# ρtrue = 0.7 et ρtrue = 0.2


# Définition du modèle
mm = MaintenanceModel(ρ=ρtrue)
gp = GammaProcess(α=αtrue,β=βtrue,θ=θtrue,mm=mm)

# Charger les données simulées
df_0_7  = CSV.read(joinpath(@__DIR__, "./Results/Convex/sim_data_full_rho0.7.csv"), DataFrame)

# Extraire les différentes tailles d'échantillons
df10_rho0_7  = df_0_7[1:10,:]
df25_rho0_7  = df_0_7[1:25,:]
df50_rho0_7  = df_0_7[1:50,:]
df75_rho0_7  = df_0_7[1:75,:]
df100_rho0_7 = df_0_7[1:100,:]
df125_rho0_7 = df_0_7[1:125,:]
df150_rho0_7 = df_0_7[1:150,:]
df175_rho0_7 = df_0_7[1:175,:]
df200_rho0_7 = df_0_7[1:200,:]

# Plot de la dégradation simulée
plot(df_0_7.tinsp, df_0_7.deg, label="Simulated degradation", xlabel="Time", ylabel="Degradation level", title="Simulated degradation process (ρ= $ρtrue)")
hline!([3], label="L")
hline!([6], label="M")

# Cas Concave : β = 0.7
# Vraies valeurs des paramètres
# αtrue = 2.0
# βtrue = 0.7
# θtrue = 2.0
# ρtrue = 0.7 et ρtrue = 0.2


# Cas Homogeneous : β=1.0
# Vraies valeurs des paramètres
# αtrue = 0.8
# βtrue = 1.0
# θtrue = 1.0
# ρtrue = 0.7 et ρtrue = 0.2

##############################################
# Analyse des résultats bayésiens
##############################################

# Cas concave et ρ=0.2 (cf. SimConcave_2.jl)
#

# MLE
df_MLE  = CSV.read(
		joinpath(@__DIR__, "Results/Concave/mle_results_rho0.2.csv"),
		DataFrame,
	)

nchains = 4

posterior_prefix = "bayes_posterior_good_priors_n10_chain"
df_4chains = [begin
	CSV.read(
		joinpath(@__DIR__, "Results/Concave/$(posterior_prefix)$(chain)_rho0.2.csv"),
		DataFrame,
	)
end for chain in 1:nchains]

res = mcmc_diagnostics(df_4chains)
display(res)

# Plot des estimations bayésiennes pour les 4 paramètres. 
# Sur chaque graphe, on représente les 4 chaines avec une couleur différente.

# Pour alpha : les quatre chaînes sur une même figure
p_alpha = plot(df_4chains[1][:, 1], label="Chaîne 1", color=:blue,
	 xlabel="Iteration", ylabel="α", title="Estimations de α")
plot!(p_alpha, df_4chains[2][:, 1], label="Chaîne 2", color=:red)
plot!(p_alpha, df_4chains[3][:, 1], label="Chaîne 3", color=:green)
plot!(p_alpha, df_4chains[4][:, 1], label="Chaîne 4", color=:orange)
display(p_alpha)

# Pour beta : 
p_beta = plot(df_4chains[1][:, 2], label="Chaîne 1", color=:blue,
	 xlabel="Iteration", ylabel="β", title="Estimations de β")
plot!(p_beta, df_4chains[2][:, 2], label="Chaîne 2", color=:red)
plot!(p_beta, df_4chains[3][:, 2], label="Chaîne 3", color=:green)
plot!(p_beta, df_4chains[4][:, 2], label="Chaîne 4", color=:orange)
display(p_beta)

# Pour theta : 
p_theta = plot(df_4chains[1][:, 3], label="Chaîne 1", color=:blue,
	 xlabel="Iteration", ylabel="θ", title="Estimations de θ")
plot!(p_theta, df_4chains[2][:, 3], label="Chaîne 2", color=:red)
plot!(p_theta, df_4chains[3][:, 3], label="Chaîne 3", color=:green)
plot!(p_theta, df_4chains[4][:, 3], label="Chaîne 4", color=:orange)
display(p_theta)

# Pour ρ : 
p_rho = plot(df_4chains[1][:, 4], label="Chaîne 1", color=:blue,
	 xlabel="Iteration", ylabel="ρ", title="Estimations de ρ")
plot!(p_rho, df_4chains[2][:, 4], label="Chaîne 2", color=:red)
plot!(p_rho, df_4chains[3][:, 4], label="Chaîne 3", color=:green)
plot!(p_rho, df_4chains[4][:, 4], label="Chaîne 4", color=:orange)
display(p_rho)


println("L'estimation bayésienne de α est $(mean(df_4chains[1][:,1]))")
println("L'estimation bayésienne de α est $(mean(df_4chains[2][:,1]))")
println("L'estimation bayésienne de α est $(mean(df_4chains[3][:,1]))")
println("L'estimation bayésienne de α est $(mean(df_4chains[4][:,1]))")



histogram(df_4chains[1][:, 1],normalize=:pdf,
    label="posterior distribution of α (good prior)",
    xlabel="α",
    ylabel="density",
    xlims=(0,6),
    ylims=(0,3))
vline!([], label="true value of alpha")
vline!([df_mle[1,1]]label="MLE of rho")
vline!([mean(df_4chains[1][:,1])],label="Bayesian Estimator of rho")



println("L'estimation bayésienne de β est $(mean(df_4chains[1][:,2]))")
println("L'estimation bayésienne de β est $(mean(df_4chains[2][:,2]))")
println("L'estimation bayésienne de β est $(mean(df_4chains[3][:,2]))")
println("L'estimation bayésienne de β est $(mean(df_4chains[4][:,2]))")

println("L'estimation bayésienne de θ est $(mean(df_4chains[1][:,3]))")
println("L'estimation bayésienne de θ est $(mean(df_4chains[2][:,3]))")
println("L'estimation bayésienne de θ est $(mean(df_4chains[3][:,3]))")
println("L'estimation bayésienne de θ est $(mean(df_4chains[4][:,3]))")

println("L'estimation bayésienne de ρ est $(mean(df_4chains[1][:,4]))")
println("L'estimation bayésienne de ρ est $(mean(df_4chains[2][:,4]))")
println("L'estimation bayésienne de ρ est $(mean(df_4chains[3][:,4]))")
println("L'estimation bayésienne de ρ est $(mean(df_4chains[4][:,4]))")






# Pour cette sortie : les R-hat (≈ 1.0003–1.0013) et les ESS (≈ 1 021–1 762)
# sont encourageants pour α, β, θ et ρ. Les taux d'acceptation rapportés pour
# α (0.253), β (0.177) et ρ (0.120) doivent s'interpréter avec les tracés et
# l'autocorrélation ; θ=missing signifie que son taux n'est pas renseigné, et
# non nécessairement qu'il est nul (à vérifier selon la mise à jour de θ).

# Cette sortie ne suffit pas, à elle seule, à confirmer que les chaînes MCMC
# ont bien convergé. Des taux d'acceptation raisonnables, un ESS suffisant
# pour la précision recherchée et des autocorrélations qui décroissent sont
# encourageants. `res` contient également le R-hat. Vérifier les tracés des
# chaînes et examiner ces diagnostics : sans leurs valeurs, impossible
# de conclure que l'échantillonnage a bien fonctionné.
# `res` contient les taux d'acceptation (`acceptance_rates`), les tailles
# effectives d'échantillon (`ess`), les autocorrélations et le R-hat. Examiner ces
# indicateurs ensemble : une autocorrélation élevée réduit l'ESS, tandis que
# les taux d'acceptation permettent d'évaluer les propositions MCMC.
# Fournir les tirages de chaque chaîne séparément sous forme de vecteur de
# DataFrame.



