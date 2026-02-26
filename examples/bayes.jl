using BayesianGammaARD
using Distributions
using DataFrames
using Statistics
using Plots

# Vraies valeurs des paramètres
αtrue = 0.8
βtrue = 1.5
θtrue = 1.0
ρtrue = 0.6

# Définition du modèle
mm = MaintenanceModel(ρ=ρtrue)
gp = GammaProcess(α=αtrue,β=βtrue,θ=θtrue,mm=mm)

HT = 20 # Fenêtre d'observation [0,T]

# Simulation du processus de dégradation
y, df = rand(gp,tinsp=1:HT,HT=HT)

# df exploitable 
mydf = BayesianGammaARD.predf(gp,df)


plot(mydf.tinsp,mydf.deg)

# Définition de la borne inf por le paramètre rho
ρlow=BayesianGammaARD.lowerboundrho(mydf)

# Initialisation du vecteur de paramètre pour l'optimisation de la fonction log-vraisemblance
x0 = [1.0, 1.5, 1.0, (1+ρlow)/2]

# Calcul du MLE
est = MLE(gp,mydf,x0,[1e-2,0.1,0.01,ρlow+0.01],[Inf, Inf, Inf, 1])

# PARTIE BAYESIENNE
# CAS Informatif
# Loi a priori pour θ (loi inverse gamma)

# Paramètre theta
priormeanθ = 2.0 # Moyenne donnée par un expert
priorvarθ = 0.5 # Variance a priori
# Calcul des paramètres de la loi inverse gamma
a = 2+ priormeanθ^2 / priorvarθ
b = (a-1) * priormeanθ
# Definition de la loi a priori pour θ
dθ = Informative(:θ,a,b,1)

# Paramètre beta 
# cas convexe (beta>1)

if βtrue>1
    priormeanβ = 1.5
    priorvarβ = 0.1

    d = priorvarβ/(priormeanβ-1)
    c = (priormeanβ-1)/d

    dβ = Informative(:β,c,d,3)
elseif βtrue==1
    pβ = 2.0
    dβ = Informative(:β,2,1,2)
else
    priormeanβ = 0.5
    priorvarβ = 0.1
    c = priormeanβ*(priormeanβ*(1-priormeanβ)/priorvarβ-1)
    d = (1-priormeanβ)*(priormeanβ*(1-priormeanβ)/priorvarβ-1)
    dβ = Informative(:β,c,d,1)
end

# Paramètre alpha (loi Gamma)
w = 1 # time given by expert à mettre à jour dans la fonction postalpha....
priormeanEw = θtrue*αtrue*w^βtrue # Degradation level at time w given by expert, ici la vraie valeur !
priorvarEw = 0.5
f = priorvarEw/priormeanEw
e = priormeanEw/f
dα = Informative(:α,e,f,2) # Le deuxième paramètre sera mis à jour dans le calcul de la post de alpha



# Paramètre rho
priormeanρ = 0.7
varpriorρ = 0.02
g = (ρlow-priormeanρ)*(ρlow-ρlow*priormeanρ-priormeanρ+priormeanρ^2+varpriorρ)/(varpriorρ*(1-ρlow))
h=-(1-priormeanρ)*(ρlow-ρlow*priormeanρ-priormeanρ+priormeanρ^2+varpriorρ)/(varpriorρ*(1-ρlow))

dρ = (1-ρlow)*Informative(:ρ,g,h,1)+ρlow

#mean(dρ)
#var(dρ)




# Cas Non Informatif




#ni = NonInformative(:θ)

#NonInformative(:θ) isa Distribution
#Uniform() isa Distribution
#NonInformative(:α) isa Distribution
#NonInformative(:β) isa Distribution

#pdf(ni,2)
#c=1.2
#d=1.5
#dd = Informative(:θ,c,d)
#params(dd)
#dd isa Distribution

#pdf(dd,2)

#ni isa Distribution
#priors = [NonInformative(:α), dβ, dθ, (1 - ρlow) * Uniform() + ρlow]
#priors = [NonInformative(:α), dβ, dθ, Uniform()]
#priors = [NonInformative(:α), NonInformative(:β), NonInformative(:θ), (1-ρlow)*Uniform()+ρlow]
#priors = [NonInformative(:α), dβ, dθ, (1 - ρlow) * Uniform() + ρlow]

priors = [dα, dβ, dθ, dρ]
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
    xlims=(0,5),
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

#priors isa Vector{ContinuousUnivariateDistribution}

# Cas Informatif homogène
## Pour θ
priormeanθ = 2
priorvarθ = 1
a = 2 + priormeanθ^2/priorvarθ
b = (a-1) * priormeanθ
dθ = Informative(:θ,a,b,2)

# Pour β proche de 1
c = 2
d = 1/c # equal to variance
dβ = Informative(:β,c,d,2)

# Pour α 




priormeanρ = ρtrue
priorvarρ = 0.0005

g = ((1-priormeanρ)*(priormeanρ-ρlow)/priorvarρ-1)*(priormeanρ - ρlow)/(1-priormeanρ)
h = (1-priormeanρ)*g/(priormeanρ-ρlow)

dρ = (1-ρlow)*Beta(g,h)+ρlow

res = algoMCMC(gp,df,[dα, dβ, dθ, dρ],10000,1.0,0.5,0.2)

##############################################
##### On fait 500 simulations pour n = 20
###############################################

αtrue = 0.8
βtrue = 1.0
θtrue = 2.0
ρtrue = 0.6
mm = MaintenanceModel(ρ=ρtrue)
gp = GammaProcess(α=αtrue,β=βtrue,θ=θtrue,mm=mm)


HT = 20
N = 500 # Nombre de simulations
estMLE = DataFrame(αMLE=zeros(N),βMLE=zeros(N),θMLE =  zeros(N),ρMLE= zeros(N))
estBayesian = DataFrame(αbayes=zeros(N),βbayes=zeros(N),θbayes =  zeros(N),ρbayes= zeros(N))

for i in 1:N
    y, df = rand(gp,tinsp=1:HT,HT=HT)
    mydf = BayesianGammaARD.predf(gp,df)

    ρlow = BayesianGammaARD.lowerboundrho(mydf)

    x0 = [1.0, 1.5, 1.0, (1+ρlow)/2]

    est = MLE(gp,mydf,x0,[1e-2,0.1,0.01,ρlow+0.01],[Inf, Inf, Inf, 1])

    priors = [NonInformative(:α), NonInformative(:β), NonInformative(:θ), (1-ρlow)*Uniform()+ρlow]
    res = algoMCMC(gp,df,priors,10000,1.0,0.5,0.2)
    res = res[1001:end,:] # On enlève les 1000 premières itérations pour éviter l'effet de démarrage

    # Stocker les résultats si nécessaire
    estMLE[i,:] = est
    estBayesian[i,:] = [mean(e) for e in eachcol(res) ]
end

MSE-MLE =    DataFrame(α=mean((estMLE.αMLE.-αtrue).^2),
    β=mean((estMLE.βMLE.-βtrue).^2),
    θ=mean((estMLE.θMLE.-θtrue).^2),
    ρ=mean((estMLE.ρMLE.-ρtrue).^2))
MSE-Bayesian = DataFrame(α=mean((estBayesian.αbayes.-αtrue).^2),
    β=mean((estBayesian.βbayes.-βtrue).^2),
    θ=mean((estBayesian.θbayes.-θtrue).^2),
    ρ=mean((estBayesian.ρbayes.-ρtrue).^2))   