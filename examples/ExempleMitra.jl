using BayesianGammaARD
using Distributions
using DataFrames
using Statistics
using Plots



αtrue = 0.8
βtrue = 2.0
θtrue = 1.5
ρtrue = 0.7
mm = MaintenanceModel(ρ=ρtrue)
gp = GammaProcess(α=αtrue,β=βtrue,θ=θtrue,mm=mm)


HT = 20 # Horizon Time
y, df = rand(gp,tinsp=1:HT,HT=HT)


ρlow=BayesianGammaARD.lowerboundrho(mydf) # borne inférieure de ρ



plot(mydf.tinsp,mydf.deg)


x0 = [1.0, 1.5, 1.0, (1+ρlow)/2]

est = MLE(gp,mydf,x0,[1e-2,0.1,0.01,ρlow+0.01],[Inf, Inf, Inf, 1])


# Choix de la loi a priori pour θ (loi inverse gamma)
priormeanθ = 2.0
priorvarθ = 0.5
a = 2+ priormeanθ^2 / priorvarθ
b = (a-1) * priormeanθ
dθ = Informative(:θ,a,b,1) # Loi Inverse Gamma pour θ

# Choix de la loi a priori pour β (loi gamma)
# Si beta proche de 1, on peut choisir une loi gamma avec une moyenne proche de 1
#pβ = 2.0
#dβ = Informative(:β,2,1,2)

# Si beta plus grand que 1, on peut choisir une loi gamma translatée

c = 1
d = 1
dβ = Informative(:β,c,d,3)
mean(dβ) # Vérification de la moyenne
var(dβ)  # Vérification de la variance

    #priors = [NonInformative(:α), dβ, dθ, (1 - ρlow) * Uniform() + ρlow]
priors = [NonInformative(:α), dβ, dθ, Uniform()]
#priors = [NonInformative(:α), NonInformative(:β), NonInformative(:θ), (1-ρlow)*Uniform()+ρlow]
res = algoMCMC(gp,df,priors,10000,1.0,0.5,0.2)
res = res[1001:end,:] # On enlève les 1000 premières itérations pour éviter l'effet de démarrage


histogram(res[:,1],normalize=:pdf,
    label="posterior distribution of alpha",
    xlabel="alpha",
    ylabel="density",
    title="Posterior distributions of parameters",
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
    xlims=(0,3),
    ylims=(0,3))
vline!([gp.β], label="true value of beta")
vline!([est[2]],label="MLE of beta")
vline!([mean(res[:,2])],label="Bayesian Estimator of beta")

histogram(res[:,3],normalize=:pdf,
    label="posterior distribution of theta",
    xlabel="theta",
    ylabel="density",
    xlims=(0,5),
    ylims=(0,3))
vline!([gp.θ], label="true value of theta")
vline!([est[3]],label="MLE of theta")
vline!([mean(res[:,3])],label="Bayesian Estimator of theta")


histogram(res[:,4],normalize=:pdf,
    label="posterior distribution of rho",
    xlabel="rho",
    ylabel="density",
    xlims=(ρlow,1),
    ylims=(0,3))
vline!([gp.mm.ρ], label="true value of alpha")
vline!([est[4]],label="MLE of rho")
vline!([mean(res[:,4])],label="Bayesian Estimator of rho")
# histogram! permet de superposer les histogrammes

