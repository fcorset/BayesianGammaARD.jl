using BayesianGammaARD
using Distributions
using DataFrames
using CSV
using Statistics
using Plots

# Vraies valeurs des paramètres
αtrue = 0.8
βtrue = 1.5
θtrue = 1.0
ρtrue = 0.7

# Définition du modèle
mm = MaintenanceModel(ρ=ρtrue)
gp = GammaProcess(α=αtrue,β=βtrue,θ=θtrue,mm=mm)


HT = 100 # Fenêtre d'observation [0,T]

# Simulation du processus de dégradation
y, df = rand(gp,tinsp=1:HT,HT=HT)

# On génère plusieurs jeux de données (n croissant):

df10 = df[1:10,:]
df20 = df[1:20,:]
df30 = df[1:30,:]
df40 = df[1:40,:]
df50 = df[1:50,:]
df60 = df[1:60,:]
df70 = df[1:70,:]
df80 = df[1:80,:]
df90 = df[1:90,:]
df100 = df[1:100,:]

# Def de ρlow pour chaque jeu de données
ρlow = Vector{Float64}(undef,10)            
EstMLE = Vector{Vector{Float64}}(undef,10)   # chaque entrée contient 4 paramètres α, β, θ, ρ

res_all = Vector{DataFrame}(undef,10)

for i in 1:length(ρlow)
    mydf = BayesianGammaARD.predf(gp, eval(Meta.parse("df" * string(i*10))))
    ρlow[i] = BayesianGammaARD.lowerboundrho(mydf)
    x0 = [1.0, 1.5, 1.0, (1+ρlow[i])/2]

    EstMLE[i] = MLE(gp, mydf, x0, [1e-2,0.1,0.01,ρlow[i]+0.01], [Inf, Inf, Inf, 1])
    # PARTIE BAYESIENNE
    # CAS Non informatif

    priors = [NonInformative(:α), NonInformative(:β), NonInformative(:θ), (1-ρlow[i])*Uniform()+ρlow[i]]
    res = algoMCMC(gp, mydf, priors, 10000, 1.0, 0.5, 0.2)
    res_all[i] = res[1001:end,:] # On enlève les 1000 premières itérations pour éviter l'effet de démarrage
end

for i in 1:length(ρlow)
    println("MLE : ", EstMLE[i])
    println("Bayesian Estimator (mean of posterior) : ", mean.(eachcol(res_all[i])))
end




mydf = BayesianGammaARD.predf(gp,df10)

plot(mydf.tinsp,mydf.deg)
hline!([3],label="L")
hline!([6],label="M")

ρlow=BayesianGammaARD.lowerboundrho(mydf)

# Initialisation du vecteur de paramètre pour l'optimisation de la fonction log-vraisemblance
x0 = [1.0, 1.5, 1.0, (1+ρlow)/2]

# Calcul du MLE
est10 = MLE(gp,mydf,x0,[1e-2,0.1,0.01,ρlow+0.01],[Inf, Inf, Inf, 1])

# PARTIE BAYESIENNE
# CAS Non informatif

priors = [NonInformative(:α), NonInformative(:β), NonInformative(:θ), (1-ρlow)*Uniform()+ρlow]

res = algoMCMC(gp,df,priors,10000,1.0,0.5,0.2)
res = res[1001:end,:] # On enlève les 1000 premières itérations pour éviter l'effet de démarrage

# Analyse des résultats Prior Non info n=10

res10_NonInfo = res

histogram(res10_NonInfo[:,1],normalize=:pdf,
    label="posterior distribution of alpha",
    xlabel="alpha",
    ylabel="density",
    title="Posterior distribution of parameter α (n=10)",
    legend=:topright,
    xlims=(0,2),
    ylims=(0,3))
vline!([gp.α],label="true value of alpha")
vline!([est[1]],label="MLE of alpha")
vline!([mean(res10_NonInfo[:,1])],label="Bayesian Estimator of alpha")

histogram(res10_NonInfo[:,2],normalize=:pdf,
    label="posterior distribution of beta",
    xlabel="beta",
    ylabel="density",
        title="Posterior distribution of parameter β (n=10)",
    xlims=(1,2.5),
    ylims=(0,3))
vline!([gp.β], label="true value of beta")
vline!([est[2]],label="MLE of beta")
vline!([mean(res10_NonInfo[:,2])],label="Bayesian Estimator of beta")

histogram(res10_NonInfo[:,3],normalize=:pdf,
    label="posterior distribution of theta",
    xlabel="theta",
    ylabel="density",
    title="Posterior distribution of parameter θ (n=10)",
    xlims=(0,5),
    ylims=(0,3))
vline!([gp.θ], label="true value of theta")
vline!([est[3]],label="MLE of theta")
vline!([mean(res10_NonInfo[:,3])],label="Bayesian Estimator of theta")

histogram(res10_NonInfo[:,4],normalize=:pdf,
    label="posterior distribution of rho",
    xlabel="rho",
    ylabel="density",
    title="Posterior distribution of parameter ρ (n=10)",
    xlims=(ρlow,1),
    ylims=(0,3))
vline!([gp.mm.ρ], label="true value of alpha")
vline!([est[4]],label="MLE of rho")
vline!([mean(res10_NonInfo[:,4])],label="Bayesian Estimator of rho")
# histogram! permet de superposer les histogrammes

# --- Sauvegarde des dataframes dans des fichiers CSV ---
CSV.write("sim_data_full.csv", df)
for i in 1:10
    subset_df = eval(Meta.parse("df" * string(i*10)))
    CSV.write("sim_data_$(i*10).csv", subset_df)
end

# Sauvegarde des résultats bayésiens (postérieurs) pour chaque jeu de données
for i in 1:length(res_all)
    CSV.write("bayes_posterior_n$(i*10).csv", res_all[i])
end

# Sauvegarde des MLE pour chaque jeu de données
mle_df = DataFrame(n = Int[], α = Float64[], β = Float64[], θ = Float64[], ρ = Float64[])
for i in 1:length(EstMLE)
    push!(mle_df, (i*10, EstMLE[i]...))
    CSV.write("mle_n$(i*10).csv", DataFrame(α = EstMLE[i][1], β = EstMLE[i][2], θ = EstMLE[i][3], ρ = EstMLE[i][4]))
end
CSV.write("mle_results.csv", mle_df)


