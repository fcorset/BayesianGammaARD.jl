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


