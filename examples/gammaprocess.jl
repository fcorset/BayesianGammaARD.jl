using BayesianGammaARD
#using DataFrames
using Statistics
using Plots
# gp = GammaProcess(β=1.5)
# GammaProcess(α=.9)

mm = MaintenanceModel(ρ=0.3)
gp = GammaProcess(α=0.8,β=2.0,θ=2.0,mm=MaintenanceModel(ρ=0.3))
y, df = rand(gp,HT=100)

mydf = BayesianGammaARD.predf(gp,df)


ρlow=BayesianGammaARD.lowerboundrho(mydf)



x0 = [1.0, 1.5, 1.0, (1+ρlow)/2]

est = MLE(gp,mydf,x0,[1e-2,0.1,0.01,ρlow+0.01],[Inf, Inf, Inf, 1])

# En simulant N trajectoires
N=1000
res=[]
for i in 1:N
    gp = GammaProcess(α=0.8,β=2,θ=2,mm=MaintenanceModel(ρ=0.7))
    y, df = rand(gp,HT=100)
    mydf = BayesianGammaARD.predf(gp,df)
    ρlow = BayesianGammaARD.lowerboundrho(mydf)
    x0 = [1.0, 1.5, 1.0, (1+ρlow)/2]

    est = MLE(gp,mydf,x0,[1e-2,0.1,0.01,ρlow+0.01],[Inf, Inf, Inf, 1])
    if i==1
        res = est
    else
        res = hcat(res,est)
    end
#    res = [res est]
end

histogram(res[1,:], label="Bayesian estimation of alpha", normalize=:pdf)
vline!([gp.α],label="true value of alpha")
title!("Histogram of alpha, 1000 samples")
histogram(res[2,:], label="Bayesian estimation of alpha", normalize=:pdf)
vline!([gp.β], label="true value of beta")
title!("Histogram of beta, 1000 samples")
histogram(res[3,:], label="Bayesian estimation of theta", normalize=:pdf)
vline!([gp.θ], label="true value of theta")
title!("Histogram of theta, 1000 samples")
histogram(res[4,:], label="Bayesian estimation of rho", normalize=:pdf)
vline!([gp.mm.ρ], label="true value of alpha")
title!("Histogram of rho, 1000 samples")



