
struct NonInformative <: ContinuousUnivariateDistribution
    pdf::Function
end

import Distributions.pdf

pdf(ni::NonInformative,x::Real)=ni.pdf(x)

const NonInformativeAlpha = NonInformative(x -> sqrt(x*trigamma(x)-1))

const NonInformativeBetaTheta = NonInformative(x -> 1/x)

function NonInformative(mode::Symbol)
    if mode==:α 
        return NonInformativeAlpha
    elseif mode==:β||mode==:θ
        NonInformativeBetaTheta
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
    for k in 1:K
        for j ∈ [3,2,1,4]
            if j==3 && priors[j] == NonInformativeBetaTheta
                
            else
 #                estpar[k,j] = rand(InverseGamma,a+??)
            end

        end

    end


end




