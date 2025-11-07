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


x0 = [1.0, 1.5, 1.0, (1+ρlow)/2]

est = MLE(gp,mydf,x0,[1e-2,0.1,0.01,ρlow+0.01],[Inf, Inf, Inf, 1])


# Choix de la loi a priori pour θ (loi inverse gamma)
priormeanθ = 2.0
priorvarθ = 0.5
a = 2+ priormeanθ^2 / priorvarθ
b = (a-1) * priormeanθ
dθ = Informative(:θ,a,b,1) # Loi Inverse Gamma pour θ

# Choix de la loi a priori pour β (loi gamma)
# 

pβ = 2.0
dβ = Informative(:β,2,1,2)

    #priors = [NonInformative(:α), dβ, dθ, (1 - ρlow) * Uniform() + ρlow]
priors = [NonInformative(:α), dβ, dθ, Uniform()]
#priors = [NonInformative(:α), NonInformative(:β), NonInformative(:θ), (1-ρlow)*Uniform()+ρlow]
res = algoMCMC(gp,df,priors,10000,1.0,0.5,0.2)
res = res[1001:end,:] # On enlève les 1000 premières itérations pour éviter l'effet de démarrage
