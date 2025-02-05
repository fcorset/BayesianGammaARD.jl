
mutable struct GammaProcess
    α::Float64 # paramètre de forme du processus Gamma η = α t^β
    β::Float64 # paramètre de forme du processus Gamma eta = α t^β
    θ::Float64 # paramètre d'échelle du processus Gamma (échelle dans Julia)
    mm::MaintenanceModel #
end

GammaProcess(; α=1.0, β=1.0, θ=1.0, mm=MaintenanceModel()) = GammaProcess(α, β, θ, mm)



import Base.rand

function rand(gp::GammaProcess; tinsp=1:100, δ::Float64=0.01, HT::Int=100)

    ts = 0:δ:HT # vecteur temps
    
    # Il faut calculer les indices tels que ts est un temps d'inspection, noté iτ
    indinsp = indexin(tinsp,ts)
    iτ = 1
    tr, ti  = 0.0, ts[indinsp[iτ]] 

    # tr : dernier temps de renouvellement
    # indinsp : vecteur d'indice des temps d'inspection
    # ti : prochain temps d'inspection
    # iτ : indice du prochain temps d'inspection dans le vecteur indinsp

    ycur=0.0 # initialisation du niveau courrant de dégradation
    y=[0.0]
    obs=[] # Observation du niveau de dégradation aux temps d'inspection
    ynew = 0.0

    for t in δ:δ:HT
        if (t-tr-δ<0)
            ycur += rand(Gamma(gp.α*((t-tr)^gp.β),gp.θ)) 
        else
            ycur += rand(Gamma(gp.α*((t-tr)^gp.β-(t-δ-tr)^gp.β),gp.θ)) 
        end
        push!(y, ycur)
        if t == ti # si le temps correspond à un temps d'inspection
            push!(obs,ycur) # On ajoute le niveau de dégradation du temps d'inspection
            ynew =  if ycur >= gp.mm.L && ycur < gp.mm.U
                        (1 - gp.mm.ρ) * ycur
            elseif ycur >= gp.mm.U
                        tr = ti # j'ai enlevé +δ (14/06/2023)
                        0.0
            else
                        ycur
            end
            if iτ < length(tinsp)
                iτ += 1
                ti = ts[indinsp[iτ]] # mise à jour du prochain temps d'inspection
            end
            ycur =  ynew
        else
            continue    
        end
    end
    df = DataFrame(tinsp = tinsp, deg = obs)
    
    # Ajout de u_{i-1} dans la dataframe
    # Ajout de degprec dans la dataframe
    return y, df
end

function predf(gp::GammaProcess, df::DataFrame)
    # Rend la df exploitable pour MLE et bayésien en calculant les u et les deg précédentes
    mydf = deepcopy(df) 
    u = Int.((mydf.deg .> gp.mm.L) .* (mydf.deg .< gp.mm.U)) # u_i dans le papier
    Δ = Int.(mydf.deg .> gp.mm.U)

    Δm1 = Δ[1:end-1]
    Δm1 = pushfirst!(Δm1,0) # Δ_{i-1} dans le papier en ajoutant zéro en 1er élément
    
    um1 = u[1:end-1]
    um1 = pushfirst!(um1,0) # u_{i-1} dans le papier en ajoutant zéro en 1er élément
    mydf[!,:Δm1] = Δm1
    mydf[!,:um1] = um1
    
    degprec = mydf.deg[1:(length(mydf.deg) - 1)]
    pushfirst!(degprec,0)
    mydf[!,:degprec] = degprec

    tr = zeros(nrow(df))
    for i in 1:nrow(df)-1
        if mydf.deg[i] > gp.mm.U  
            tr[i+1:nrow(df)] .= mydf.tinsp[i]
        end
    end
    s = mydf.tinsp .-tr
    spre = [0;mydf.tinsp[1:nrow(df)-1]] .- tr
    mydf[!,:s] = s
    mydf[!,:spre] = spre
    mydf[!,:tr] = tr        
    return mydf
end

function deltaeta(mydf::DataFrame,α,β)
    return α .* (mydf.s.^β - mydf.spre.^β)
end




import Distributions.loglikelihood

function loglikelihood(gp::GammaProcess,x::Vector{Float64},mydf::DataFrame)
    part1 = (x[1] * mydf.tinsp[1]^x[2] - 1) * log(mydf.deg[1]) - log(x[3]) * x[1] * mydf.tinsp[1]^x[2] - log(gamma(x[1] * mydf.tinsp[1]^x[2])) +  - mydf.deg[1]/x[3] # part de la vraisemenblance où u_{i-1}=0.
        tr = 0 # dernier temps de renouvellement
        for i in 2:nrow(mydf)
            Δηi = x[1] * ((mydf.tinsp[i] - tr)^x[2] - (mydf.tinsp[i-1] - tr)^x[2])
            if mydf.um1[i] == 0
                part1 += if mydf.deg[i-1] > gp.mm.U
                    #if Δηi<0
                    #    println(Δηi)
                    #end
                    #if mydf.deg[i]<0
                    #    println(mydf.deg[i])
                    #end
                    (Δηi - 1) * log(mydf.deg[i]) - Δηi*log(x[3])  - log(gamma(Δηi))  - mydf.deg[i]/x[3]
                else
                    (Δηi - 1) * log(mydf.deg[i] - mydf.deg[i-1])  - log(x[3]) * Δηi - log(gamma(Δηi)) - (mydf.deg[i] - mydf.deg[i-1])/x[3]
                end
            else
                (Δηi - 1) * log(mydf.deg[i] - (1-x[4])*mydf.deg[i-1])  - log(x[3]) * Δηi - log(gamma(Δηi)) - (mydf.deg[i] - (1-x[4])*mydf.deg[i-1])/x[3]
            end
            if mydf.deg[i] > gp.mm.U
                tr = mydf.tinsp[i] # On met à jour le dernier instant de renouvellement
            end        
        end
    return -part1  
end

function MLE(gp::GammaProcess,mydf::DataFrame,x0 = [1.0, 1.5, 1.0, 0.8],lower = [1e-2,0.1,0.01,0],upper=[Inf, Inf, Inf, 1])
    myLogL(x) = loglikelihood(gp,x,mydf)
    res = optimize(myLogL,lower, upper, x0)
    res2 = Optim.minimizer(res)
    return res2
end


function lowerboundrho(df::DataFrame)
    mydf=filter(row -> row.um1 ==1, df)
    return maximum([0;1 .- mydf.deg ./ mydf.degprec])
end
