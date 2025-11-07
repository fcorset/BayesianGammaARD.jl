using BayesianGammaARD
using Distributions
using DataFrames
using Statistics
using Plots

αtrue = 0.8
βtrue = 1.0
θtrue = 2.0
ρtrue = 0.6
mm = MaintenanceModel(ρ=ρtrue)
gp = GammaProcess(α=αtrue,β=βtrue,θ=θtrue,mm=mm)


HT = 20
y, df = rand(gp,tinsp=1:HT,HT=HT)


mydf = BayesianGammaARD.predf(gp,df)

plot(mydf.tinsp,mydf.deg)

ρlow=BayesianGammaARD.lowerboundrho(mydf)


x0 = [1.0, 1.5, 1.0, (1+ρlow)/2]

est = MLE(gp,mydf,x0,[1e-2,0.1,0.01,ρlow+0.01],[Inf, Inf, Inf, 1])


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
priormeanθ = 2.0
    priorvarθ = 0.5
    a = 2+ priormeanθ^2 / priorvarθ
    b = (a-1) * priormeanθ
    dθ = Informative(:θ,a,b,1)

    pβ = 2.0
    dβ = Informative(:β,2,1,2)

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
    xlims=(0,2),
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
w = 1 # time given by expert
priormeanEw = θtrue*αEtrue # Degradation level at time w given by expert, ici la vraie valeur !
priorvarEw = 1
f = priorvarEw/priormeanw
e = priormeanEw/f
dα = Informative(:α,e,f,2) # A changer... le deuxième paramètre dépend de θ et β !!!!




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