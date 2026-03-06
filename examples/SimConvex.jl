using BayesianGammaARD
using Distributions
using DataFrames
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

# df exploitable 
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


histogram(res10_NonInfo [:,1],normalize=:pdf,
    label="posterior distribution of alpha",
    xlabel="alpha",
    ylabel="density",
    title="Posterior distribution of parameter α (n=10)",
    legend=:topright,
    xlims=(0,2),
    ylims=(0,3))
vline!([gp.α],label="true value of alpha")
vline!([est[1]],label="MLE of alpha")
vline!([mean(res[:,1])],label="Bayesian Estimator of alpha")

histogram(res[:,2],normalize=:pdf,
    label="posterior distribution of beta",
    xlabel="beta",
    ylabel="density",
        title="Posterior distribution of parameter β (n=10)",
    xlims=(0,5),
    ylims=(0,3))
vline!([gp.β], label="true value of beta")
vline!([est[2]],label="MLE of beta")
vline!([mean(res[:,2])],label="Bayesian Estimator of beta")

histogram(res[:,3],normalize=:pdf,
    label="posterior distribution of theta",
    xlabel="theta",
    ylabel="density",
    title="Posterior distribution of parameter θ (n=10)",
    xlims=(0,5),
    ylims=(0,3))
vline!([gp.θ], label="true value of theta")
vline!([est[3]],label="MLE of theta")
vline!([mean(res[:,3])],label="Bayesian Estimator of theta")
histogram(res[:,4],normalize=:pdf,
    label="posterior distribution of rho",
    xlabel="rho",
    ylabel="density",
    title="Posterior distribution of parameter ρ (n=10)",
    xlims=(ρlow,1),
    ylims=(0,3))
vline!([gp.mm.ρ], label="true value of alpha")
vline!([est[4]],label="MLE of rho")
vline!([mean(res[:,4])],label="Bayesian Estimator of rho")
# histogram! permet de superposer les histogrammes



