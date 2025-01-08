using Distributions
using Plots
using Optim
using DataFrames
using Statistics
using SpecialFunctions
using CSV

mutable struct GammaProcess
    α::Float64 # paramètre de forme du processus Gamma eta = α t^β
    β::Float64 # paramètre de forme du processus Gamma eta = α t^β
    θ::Float64 # paramètre d'échelle du processus Gamma (échelle dans Julia)
end


mutable struct MaintenanceModels
    L::Float64 # seuil L à partir duquel on déclenche une Maintenance Préventive (ARD)
    M::Float64 # seuil M à partir duquel on déclenche une Maintenance Corrective (AGAN)
    τ::Float64 # temps inter-inspection

end


import Base.rand

function rand(gm::GammaM; τ::Vector{Int}=sort(rand(1:10001,10)), δ::Float64=0.01, HT::Int=100.0)
    
    # τ : indice du vecteur temps d'inspections
    # ts[τ] : Temps d'inspection
    # δ : pas de simulation du processus gamma


    ts = 0:δ:HT # vecteur temps
    tr, iτ, ti  = 0.0, 1, ts[τ[1]] 
    # tr : dernier temps de renouvellement
    # iτ : indice du vecteur temps d'inspection
    # ti : indice du prochain temps d'inspection
    

    ycur=0.0 # initialisation du niveau courrant de dégradation
    y=[0.0]
    obs=[] # Observation du niveau de dégradation aux temps d'inspection
    vectb=[] # Vecteur indicatrice maintenance efficace
    ynew = 0.0

    for t in δ:δ:HT
        ycur += rand(Gamma(gm.α*((t-tr)^gm.β-(t-δ-tr)^gm.β),1/gm.b)) # On simule l'incrément de gamma
        push!(y, ycur)
        if t == ti # si le temps correspond à un temps d'inspection
            push!(obs,ycur) # On ajoute le niveau de dégradation du temps d'inspection

            ynew =  if ycur >= gm.L && ycur < gm.M
               if rand() < gm.p 
                    push!(vectb,1)
                    (1 - gm.ρ) * ycur
                else
                    push!(vectb,0)
                    (1 + gm.ρw) * ycur
                end 
            elseif ycur >= gm.M
                push!(vectb,1)
                tr = ti # j'ai enlevé +δ (14/06/2023)
                0.0
            else
                push!(vectb,1)
                ycur
            end
            if iτ < length(τ)
                iτ += 1
                ti = ts[τ[iτ]] # mise à jour du prochain temps d'inspection
            end
            
            ycur =  ynew
            #if  ynew > 0.0
            #    push!(y, ycur)
            #    continue
            #end
        else
            continue    
        end
        
    end
    df = DataFrame(tinsp = ts[τ], deg = obs, indeff = vectb)
    return y, df

end


α = 1.0
β = 1.0
b = 1.0
ρ = 0.8
ρw = 0.3
p = 0.9
L = 5.0 # seuil de MP
M = 10.0 # seuil de MC
HT = 100
δ = 0.01
τ = collect((100+1):100:(HT*100+1))

Nsim = 500 # Nb de simulations

estα = []
estβ = []
estb = []
estρ = []
estρw = []
estp = []




gm = GammaM(α,β,b,ρ,ρw,p,L,M)
for j in 1:Nsim
    # show(j)
    y, mydf = rand(gm,τ=τ,HT=HT)
    tmp = "./DATA/df_$(α)_$(β)_$(b)_$(ρ)_$(ρw)_$(p)_$j.csv"
    CSV.write(tmp,mydf)
    # Calcul des u
    u = Int.((mydf.deg .> L) .* (mydf.deg .< M)) # u_i dans le papier
    um1 = u[1:(length(u)-1)]
    um1 = pushfirst!(um1,0) # u_{i-1} dans le papier en ajoutant zéro en 1er élément
    # Ajout de u_{i-1} dans la dataframe
    mydf[!,:um1] = um1
    # Ajout de degprec dans la dataframe
    degprec = mydf.deg[1:(length(mydf.deg) - 1)]
    pushfirst!(degprec,0)
    mydf[!,:degprec] = degprec

    function EspLogLikabc(x)
        part1 = log(x[3]) * x[1] * mydf.tinsp[1]^x[2] - log(gamma(x[1] * mydf.tinsp[1]^x[2])) + (x[1] * mydf.tinsp[1]^x[2] - 1) * log(mydf.deg[1]) - x[3] * mydf.deg[1] # part de la vraisemenblance où u_{i-1}=0.
        tr = 0 # dernier temps de renouvellement
        for i in 2:nrow(mydf)
            Δai = x[1] * ((mydf.tinsp[i] - tr)^x[2] - (mydf.tinsp[i-1] - tr)^x[2])
            if um1[i] == 0
                part1 += if mydf.deg[i-1] > M
                    log(x[3]) * Δai - log(gamma(Δai)) + (Δai - 1) * log(mydf.deg[i]) - x[3] * mydf.deg[i]
                else
                    log(x[3]) * Δai - log(gamma(Δai)) + (Δai - 1) * log(mydf.deg[i] - mydf.deg[i-1]) - x[3] * (mydf.deg[i] - mydf.deg[i-1])
                end
            end
            if mydf.deg[i] > M
                tr = mydf.tinsp[i] # On met à jour le dernier instant de renouvellement
            end        
        end
        return -part1
    end

    x0 = [1.0, 1.5, 1.0]
    res = optimize(EspLogLikabc, x0)
    res2 = Optim.minimizer(res)
    αhat = res2[1]
    βhat = res2[2]
    bhat = res2[3]
    push!(estα,αhat)
    push!(estβ,βhat)
    push!(estb,bhat)

    function EspLogLikrho(x)
        tr = 0
        res = 0
        for i in 2:nrow(mydf)
            if (mydf.deg[i] < mydf.deg[i-1]) & (um1[i] == 1)
                Δai = αhat * (mydf.tinsp[i]^βhat - mydf.tinsp[i-1]^βhat)
                res += (Δai - 1) * log(mydf.deg[i] - (1 - x) * mydf.deg[i-1]) - bhat * (mydf.deg[i] - (1 - x) * mydf.deg[i-1])
            end
        end
        return - res
    end
    
    x0 = 0.5
    # Définir la borne inf de l'intervalle sur lequel on optimise la fonction
    mydf1 = subset(mydf,:um1 => ByRow(>(0)))
    
    ρlow = maximum(1 .- mydf1.deg ./ mydf1.degprec)
    res = optimize(EspLogLikrho, ρlow, 1)
    ρhat = Optim.minimizer(res)

    push!(estρ,ρhat)
    
    function AlgoEM(K::Int=50)
        # K : Nb itérations AlgoEM
        ρwhat = [0.2] # initialisation de ρw
        phat = [0.7] # Initialisation de p
        # On ne garde que les y_i tq u_{i-1}=1
        ptilde = [] #ones(nrow(mydf1))
        for i in 1:nrow(mydf1)
            if mydf1.deg[i] > mydf1.degprec[i]
                Δai  = αhat * (mydf1.tinsp[i]^βhat - (mydf1.tinsp[i] - 1)^βhat)
                push!(ptilde,phat[1] * pdf(Gamma(Δai,1/bhat),mydf1.deg[i] - (1-ρhat) * mydf1.degprec[i])/(phat[1] * pdf(Gamma(Δai,1/bhat),mydf1.deg[i] - (1-ρhat[1]) * mydf1.degprec[i])+(1 - phat[1])* pdf(Gamma(Δai,1/bhat),mydf1.deg[i] - (1+ρwhat[1]) * mydf1.degprec[i])))
            else
                push!(ptilde,1.0)
            end
        end
        
        mydf1[!,:ptilde] = ptilde
        mydftemp = subset(mydf1,:ptilde => ByRow(<(1)))
    
        
    
        for k in 2:(K+1)
            push!(phat,mean(ptilde))
            if nrow(mydftemp) > 0
                ρwupper = minimum(mydftemp.deg./mydftemp.degprec.-1)
                function EspLogLikrhow(x)
                    res = 0
                    for i in 1:nrow(mydftemp)
                        Δai = αhat * (mydftemp.tinsp[i]^βhat - (mydftemp.tinsp[i] - 1)^βhat)
                        res += mydftemp.ptilde[i] * (Δai - 1) * log(mydftemp.deg[i] - (1 - ρhat) * mydftemp.degprec[i]) + (1 - mydftemp.ptilde[i]) * (Δai - 1) * log(mydftemp.deg[i] - (1 + x) * mydftemp.degprec[i]) - b * (mydftemp.ptilde[i] * (mydftemp.deg[i] - (1 - ρhat) * mydftemp.degprec[i]) + (1 - mydftemp.ptilde[i]) * (mydftemp.deg[i] - (1 + x) * mydftemp.degprec[i]))
                    end
                return -res
                end
                res = optimize(EspLogLikrhow, 0, ρwupper)
                push!(ρwhat, Optim.minimizer(res))
            else
                push!(ρwhat, - 1)
            end
            
            # Mise à jour des ptilde
            ptilde = [] #ones(nrow(mydf1))
            for i in 1:nrow(mydf1)
                if mydf1.deg[i] > mydf1.degprec[i]
                    Δai  = αhat * (mydf1.tinsp[i]^βhat - (mydf1.tinsp[i] - 1)^βhat)
                    push!(ptilde,phat[k] * pdf(Gamma(Δai,1/bhat),mydf1.deg[i] - (1-ρhat) * mydf1.degprec[i])/(phat[k] * pdf(Gamma(Δai,1/bhat),mydf1.deg[i] - (1-ρhat[1]) * mydf1.degprec[i])+(1 - phat[k])* pdf(Gamma(Δai,1/bhat),mydf1.deg[i] - (1+ρwhat[k]) * mydf1.degprec[i])))
                else
                    push!(ptilde,1.0)
                end
            end
            mydf1[!,:ptilde] = ptilde
            mydftemp = subset(mydf1,:ptilde => ByRow(<(1)))
    
        end
        return ρwhat, phat 
    end
    K = 50
    res = AlgoEM(K)
    ρwhat = res[1][K+1]
    phat = res[2][K+1]
    push!(estρw,ρwhat)
    push!(estp,phat)
end
mytxt = "./Results/res_$(α)_$(β)_$(b)_$(ρ)_$(ρw)_$(p).csv"
CSV.write(mytxt, DataFrame(estα = estα, estβ = estβ, estb = estb, estρ = estρ, estρw = estρw, estp = estp))


# TO DO 
# Exlure les cas où on estime à $1$ le paramètre p, car on ne peut pas estimer ρw 
# Vérifier dans ces cas là, s'il y a ou non des maintenances nuisibles