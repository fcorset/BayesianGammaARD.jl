
struct NonInformative <: ContinuousUnivariateDistribution
    pdf::Function
end

struct Informative <: ContinuousUnivariateDistribution
    pdf::Function
end

import Distributions.pdf

pdf(ni::NonInformative,x::Real)=ni.pdf(x)
pdf(info::Informative,x::Real)=info.pdf(x)

const NonInformativeAlpha = NonInformative(x -> sqrt(x*trigamma(x)-1))
const InformativeBetaInf1 = Beta
const InformativeBetaEq1 = Gamma # with d=1/c
const InformativeBetaSup1 = Gamma
const InformativeTheta = InverseGamma


const NonInformativeBetaTheta = NonInformative(x -> 1/x)

function NonInformative(mode::Symbol)
    if mode==:α 
        return NonInformativeAlpha
    elseif mode==:β||mode==:θ
        NonInformativeBetaTheta
    end
end

function Informative(mode::Symbol,par1,par2)
    if mode==:θ
        InverseGamma(par1,par2)
    elseif mode==:β
        Gamma(par1,par2)
    elseif mode==:ρ
        Beta(par1,par2)
    end
end





function logcondpostdistbeta(β,priors::Vector{Distribution},mydf::DataFrame,θ,α,ρ)
    for j in 1:nrow(mydf)
        sum((deltaeta(mydf,α,β)-1).*log.(mydf.deg .- exp.(mydf.um1 .* log(1-ρ)) .* mydf.Δm1 .* mydf.degprec))-log(θ)*sum(deltaeta(mydf,α,β))-sum(log(gamma(deltaeta(mydf,α,β)))) + log(priors[2](β))
    end
end



# Faire une fonction qui prend une DataFrame (données) et les loi a priori
# Renvoie un échantillon des paramètres après un algo MCMC (Gibbs + MH)

function algoMCMC(gp::GammaProcess,df::DataFrame, priors::Vector{Distribution}=[NonInformative(:α), NonInformative(:β), NonInformative(:θ), Uniform()], K::Int=10000)
    # Choisir des valeurs initiales pour les paramètres (on prend le MLE)
    mydf = predf(gp::GammaProcess, df::DataFrame)
    lbrho = lowerboundrho(mydf)
    parinit = MLE(gp,mydf,x0 = [1.0, 1.5, 1.0, (1+lbrho)/2],lower = [1e-2,0.1,0.01,lbrho+0.01],upper=[Inf, Inf, Inf, 1])
    
    estpar::DataFrame(α=zeros(K+1),β=zeros(K+1),θ=zeros(K+1),ρ=zeros(K+1))
    estpar[1,:]=parinit
    for k in 2:K+1
        for j ∈ [3,2,1,4]
            if j==3 && priors[j] == NonInformativeBetaTheta
                estpar[k,j] = rand(InverseGamma,sum(deltaeta(mydf,estpar[k-1,1],estpar[k-1,2])),sum(mydf.deg .- exp.(mydf.um1 .* log(1-estpar[k-1,4])) .* mydf.Δm1 .* mydf.degprec))
            else
                estpar[k,j] = rand(InverseGamma,a+sum(deltaeta(mydf,estpar[k-1,1],estpar[k-1,2])),b+sum(mydf.deg .- exp.(mydf.um1 .* log(1-estpar[k-1,4])) .* mydf.Δm1 .* mydf.degprec))
            end
            if j==2
                if priors[j] == InformativeBetaInf1

                elseif priors[j] == GammaInformativeBetaEq1
                
                elseif priors[j] == InformativeBetaSup1
                
                else
                    # Cas Non Informatif
                    βstar = rand(Normal(estpar[k-1,2],sqrt(tau2)))
                    logratio  = logcondpostdistbeta(βstar,priors,mydf,estpar[k,3],estpar[k-1,1],estpar[k-1,4])-logcondpostdistbeta(estpar[k-1,2],priors,mydf,estpar[k,3],estpar[k-1,1],estpar[k-1,4])
                    if logratio < rand(Uniform())
                        estpar[k,2] = βstar
                    else
                        estpar[k,2] = estpar[k-1,2]
                    end
                end
            # Différencier les cas β<1, β=1 et β>1
            end

        end

    end


end




