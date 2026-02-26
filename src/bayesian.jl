import Distributions.pdf

struct NonInformative <: ContinuousUnivariateDistribution
    pdf::Function
end

struct Informative <: ContinuousUnivariateDistribution
    pdf::Function
end




pdf(ni::NonInformative,x::Real)=ni.pdf(x)
pdf(info::Informative,x::Real)=info.pdf(x)

const NonInformativeAlpha = NonInformative(x -> sqrt(x*trigamma(x)-1))
const NonInformativeBetaTheta = NonInformative(x -> if x>0 return 1/x else return 0 end)
const NonInformativeRho = NonInformative(x -> if x>=0 && x <= 1 return 1 else return 0 end)

const InformativeBetaInf1 = Beta
const InformativeBetaEq1 = Gamma # with d=1/c
const InformativeBetaSup1 = Gamma
const InformativeTheta = InverseGamma



function NonInformative(mode::Symbol)
    if mode==:α 
        return NonInformativeAlpha
    elseif mode==:β||mode==:θ
        return NonInformativeBetaTheta
    elseif mode==:ρ
        return NonInformativeRho
    end
end

function Informative(mode::Symbol,par1,par2,par3)
    # par3 représente le mode de la loi a priori de β
    # =1 si β<1
    # =2 si β=1
    # =3 si β>1
    if mode==:θ
        InverseGamma(par1,par2)
    elseif mode==:β
        if par3==1
            Beta(par1,par2)
        elseif par3==2      
            Gamma(par1,1/par1)
        elseif par3==3
            1+Gamma(par1,par2)
        end
    elseif mode==:ρ
        Beta(par1,par2)
    elseif mode==:α
        Gamma(par1,par2)
    end
end


function logcondpostdistbeta(β,priors::Vector{ContinuousUnivariateDistribution},mydf::DataFrame,θ,α,ρ,lbrho)
    if β <=0
        return -Inf
    else
        s = 0
        for i in 1:nrow(mydf)
            s += (deltaeta(mydf,α,β)[i]-1)*log(mydf.deg[i] - exp(mydf.um1[i] * log(1-ρ)) * (1 - mydf.Δm1[i]) * mydf.degprec[i]) -log(θ)*deltaeta(mydf,α,β)[i] - log(gamma(deltaeta(mydf,α,β)[i]))
        end
        dd = priors[2]
        s+=log(pdf(dd,β))    
        return s
    end
end

function logcondpostdistalpha(α,priors::Vector{ContinuousUnivariateDistribution},mydf::DataFrame,β,θ,ρ,lbrho)
    if α <= 0
        return -Inf
    else
        s = 0
        for i in 1:nrow(mydf)
            s+= (deltaeta(mydf,α,β)[i]-1)*log(mydf.deg[i] - exp(mydf.um1[i] * log(1-ρ)) * (1 - mydf.Δm1[i]) * mydf.degprec[i]) -log(θ)*deltaeta(mydf,α,β)[i] - log(gamma(deltaeta(mydf,α,β)[i]))
        end
        dd=priors[1]
        if dd==NonInformativeAlpha
            s+=log(pdf(dd,α))
        else
            w=1 # A changer...
            s+=log(pdf(dd,α*θ*w^β))+log(θ*w^β)

#            s+=log(pdf(dd,α * θ * w^β))
            # Il faut donner w en argument !!!
            s+=log(pdf(dd,α * θ))
        end
        return s
    end
end

function logcondpostdistrho(ρ,priors::Vector{ContinuousUnivariateDistribution},mydf::DataFrame,α,β,θ,lbrho)
    dd=priors[4]
    if ρ <= lbrho || ρ >= 1
        return -Inf
    else
        return sum((deltaeta(mydf,α,β).-1).*log.(mydf.deg .- exp.(mydf.um1 .* log(1-ρ)) .* (1 .- mydf.Δm1) .* mydf.degprec)) - sum(mydf.deg .- exp.(mydf.um1 .* log(1-ρ)) .* (1 .- mydf.Δm1) .* mydf.degprec)/θ+ log(pdf(dd,ρ))
    end    
end


# Faire une fonction qui prend une DataFrame (données) et les loi a priori
# Renvoie un échantillon des paramètres après un algo MCMC (Gibbs + MH)

function algoMCMC(gp::GammaProcess,df::DataFrame, priors::Vector{ContinuousUnivariateDistribution}=[NonInformative(:α), NonInformative(:β), NonInformative(:θ), Uniform()], K::Int=10000,tau2α::Float64=1.0,tau2β::Float64=0.5,tau2ρ::Float64=0.2)
    # Choisir des valeurs initiales pour les paramètres (on prend le MLE)
    dα = priors[1]
    dβ = priors[2]
    dθ = priors[3]
    dρ = priors[4]   
    mydf = predf(gp::GammaProcess, df::DataFrame)
    lbrho = lowerboundrho(mydf)
    #parinit = MLE(gp,mydf,x0 = [1.0, 1.5, 1.0, (1+lbrho)/2],lower = [1e-2,0.1,0.01,lbrho+0.01],upper=[Inf, Inf, Inf, 1])
    parinit = [1.0, 1.5, 1.0, (1+lbrho)/2]
    estpar = DataFrame((α=zeros(K+1),β=zeros(K+1),θ=zeros(K+1),ρ=zeros(K+1)))
    estpar[1,:]=parinit
    for k in 2:K+1
#        println("La valeur de k est ",k)
        for j ∈ [3,2,1,4]
            if j==3 
                if priors[j] == NonInformativeBetaTheta
                    estpar[k,j] = rand(InverseGamma(sum(deltaeta(mydf,estpar[k-1,1],estpar[k-1,2])), sum(mydf.deg .- exp.(mydf.um1 .* log(1-estpar[k-1,4])) .* (1 .- mydf.Δm1) .* mydf.degprec)))
                else
                    estpar[k,j] = rand(InverseGamma(params(dθ)[1]+sum(deltaeta(mydf,estpar[k-1,1],estpar[k-1,2])),params(dθ)[2]+sum(mydf.deg .- exp.(mydf.um1 .* log(1-estpar[k-1,4])) .* (1 .- mydf.Δm1) .* mydf.degprec)))
                end
            end
            if j==2
                if priors[j] == InformativeBetaInf1

                elseif priors[j] == InformativeBetaEq1
                
                elseif priors[j] == InformativeBetaSup1
                
                else
                    # Cas Non Informatif
                    βstar = rand(Normal(estpar[k-1,2],sqrt(tau2β)))
#                    println(βstar)
                    logratio  = logcondpostdistbeta(βstar,priors,mydf,estpar[k,3],estpar[k-1,1],estpar[k-1,4],lbrho)-logcondpostdistbeta(estpar[k-1,2],priors,mydf,estpar[k,3],estpar[k-1,1],estpar[k-1,4],lbrho)
                    if logratio > log(rand(Uniform()))
                        estpar[k,2] = βstar
                    else
                        estpar[k,2] = estpar[k-1,2]
                    end
                end
            # Différencier les cas β<1, β=1 et β>1
            end
            if j==1 
                if priors[j] == NonInformativeAlpha
                    αstar = rand(Normal(estpar[k-1,1],sqrt(tau2α)))
                    logratio  = logcondpostdistalpha(αstar,priors,mydf,estpar[k,2],estpar[k,3],estpar[k-1,4],lbrho)-logcondpostdistalpha(estpar[k-1,1],priors,mydf,estpar[k,2],estpar[k,3],estpar[k-1,4],lbrho)
                    if logratio > log(rand(Uniform()))
                        estpar[k,1] = αstar
                    else
                        estpar[k,1] = estpar[k-1,1]
                    end
                else
                    αstar = rand(Normal(estpar[k-1,1],sqrt(tau2α)))
                    logratio  = logcondpostdistalpha(αstar,priors,mydf,estpar[k,2],estpar[k,3],estpar[k-1,4],lbrho)-logcondpostdistalpha(estpar[k-1,1],priors,mydf,estpar[k,2],estpar[k,3],estpar[k-1,4],lbrho)
                    if logratio > log(rand(Uniform()))
                        estpar[k,1] = αstar
                    else
                        estpar[k,1] = estpar[k-1,1]
                    end    
                end
            end
            if j==4
                if priors[j] == Uniform()
                    ρstar = rand(Normal(estpar[k-1,4],sqrt(tau2ρ)))
                    logratio  = logcondpostdistrho(ρstar,priors,mydf,estpar[k,1],estpar[k,2],estpar[k,3],lbrho)-logcondpostdistrho(estpar[k-1,4],priors,mydf,estpar[k,1],estpar[k,2],estpar[k,3],lbrho)
                    if logratio > log(rand(Uniform()))
                        estpar[k,4] = ρstar
                    else
                        estpar[k,4] = estpar[k-1,4]
                    end
                else
                    estpar[k,j] = rand(priors[j])
                end
            end
        end
#        println(estpar[k,:])
    end
    return estpar
end




