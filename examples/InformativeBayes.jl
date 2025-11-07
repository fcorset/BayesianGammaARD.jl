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


HT = 10
N = 500
estMLE = DataFrame(αMLE=zeros(N),βMLE=zeros(N),θMLE =  zeros(N),ρMLE= zeros(N))
estBayesian = DataFrame(αbayes=zeros(N),βbayes=zeros(N),θbayes =  zeros(N),ρbayes= zeros(N))








for i in 1:N
    y, df = rand(gp,tinsp=1:HT,HT=HT)
    mydf = BayesianGammaARD.predf(gp,df)

    ρlow = BayesianGammaARD.lowerboundrho(mydf)

    x0 = [1.0, 1.5, 1.0, (1 + ρlow) / 2]

    est = MLE(gp,mydf,x0,[1e-2,0.1,0.01,ρlow+0.01],[Inf, Inf, Inf, 1])

    priormeanθ = 2.0
    priorvarθ = 0.5
    a = 2+ priormeanθ^2 / priorvarθ
    b = (a-1) * priormeanθ
    dθ = Informative(:θ,a,b,1)

    pβ = 2.0
    dβ = Informative(:β,2,1,2)

#    priors = [NonInformative(:α), dβ, dθ, (1 - ρlow) * Uniform() + ρlow]
   priors = [NonInformative(:α), dβ, dθ, Uniform()]
 
#priors = [NonInformative(:α), dβ, NonInformative(:θ), (1 - ρlow) * Uniform() + ρlow]
    res = algoMCMC(gp,df,priors,10000,1.0,0.5,0.2)
    res = res[1001:end,:] # On enlève les 1000 premières itérations pour éviter l'effet de démarrage
 # Stocker les résultats si nécessaire
    estMLE[i,:] = est
    estBayesian[i,:] = [mean(e) for e in eachcol(res) ]
end

MSEαMLE = mean((estMLE[:,1].-αtrue).^2)
MSEBayesianα = mean((estBayesian[:,1].-αtrue).^2)

MSEβMLE = mean((estMLE[:,2].-βtrue).^2)
MSEBayesianβ = mean((estBayesian[:,2].-βtrue).^2)

MSEθMLE = mean((estMLE[:,3].-θtrue).^2)
MSEBayesianθ = mean((estBayesian[:,3].-θtrue).^2)

MSEρMLE = mean((estMLE[:,4].-ρtrue).^2)
MSEBayesianρ = mean((estBayesian[:,4].-ρtrue).^2)
using Latexify

mse_table = DataFrame(
    Method = ["MLE", "Bayesian"],
    α = [MSEαMLE, MSEBayesianα],
    β = [MSEβMLE, MSEBayesianβ],
    θ = [MSEθMLE, MSEBayesianθ],
    ρ = [MSEρMLE, MSEBayesianρ]
)

println(latexify(mse_table, env=:table))