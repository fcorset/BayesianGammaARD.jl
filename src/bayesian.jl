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

# Alias non casse-sensitive pour compatibilité (potentiellement `Noninformative` utilisé par erreur)
const Noninformative = NonInformative   

const InformativeBetaInf1 = Beta
const InformativeBetaEq1 = Gamma # with d=1/c
const InformativeBetaSup1 = Gamma
const InformativeTheta = InverseGamma

"""
    simuler_une_gig(p, a, b) -> Float64

Génère une unique valeur aléatoire (scalaire) suivant la loi GIG(p, a, b).
Paramètres :
- p : Paramètre de forme (lambda), peut être négatif.
- a : Paramètre chi (> 0).
- b : Paramètre psi (> 0).
"""
function rand_gig(p::Real, a::Real, b::Real)::Float64
    # Conversion et gestion de la symétrie de la GIG si p < 0
    _p, _a, _b = Float64(p), Float64(a), Float64(b)
    inverse = _p < 0.0
    if inverse
        _p = -_p
        _a, _b = _b, _a
    end

    # Algorithme de rejet adaptatif (Dagpunar)
    # 1. Calcul du mode de la distribution
    mode = ((_p - 1.0) + sqrt((_p - 1.0)^2 + _a * _b)) / _b
    
    alpha = 0.5 * _b
    beta = 0.5 * _a
    g_x0 = (_p - 1.0) * log(mode) - alpha * mode - beta / mode
    
    # Paramètre de l'enveloppe exponentielle
    lam = alpha + beta / (mode^2)
    
    while true
        u = rand()
        v = rand()
        
        # Génération du candidat x (Loi instrumentale bi-exponentielle)
        x = u <= 0.5 ? mode + log(v) / lam : mode - log(v) / lam
        
        x <= 0.0 && continue
        
        # Critère d'acceptation
        g_x = (_p - 1.0) * log(x) - alpha * x - beta / x
        
        if log(rand()) <= (g_x - g_x0 + lam * abs(x - mode))
            return inverse ? 1.0 / x : x
        end
    end
end

# --- Exemple d'utilisation ---
#valeur = simuler_une_gig(0.5, 2.0, 1.5)
#println("Valeur simulée unique : ", valeur)


"""
function rand_gig(p::Real, a::Real, b::Real; n::Union{Nothing,Integer}=nothing,rng::AbstractRNG=Random.default_rng())
    a > 0 || throw(ArgumentError("a doit être strictement positif."))
    b > 0 || throw(ArgumentError("b doit être strictement positif."))

    draw() = begin
        ω = sqrt(a * b)
        scale = sqrt(a / b)

        # Cas particuliers exacts
        if p == 0.5
            μ = sqrt(a / b)
            shape = a
            return rand(rng, InverseGaussian(μ, shape))
        elseif p == -0.5
            μ = sqrt(b / a)
            shape = b
            return 1 / rand(rng, InverseGaussian(μ, shape))
        end

        α = abs(p)
        z = ω
        root = sqrt(z^2 + 4z * α)

        t = (1 + α + root) / z
        m = 1 / t

        log_m_const = -α * log(t) + 0.5 * log(t) -
                      z * (t - 1 / t) / 2

        while true
            # Proposition inverse gaussienne
            y = rand(rng, InverseGaussian(m, 1.0))

            log_accept = (α - 0.5) * log(y) -
                         z * (y + 1 / y) / 2 -
                         log_m_const

            if log(rand(rng)) <= log_accept
                v = p > 0 ? y : 1 / y
                return scale * v
            end
        end
    end

    if isnothing(n)
        return draw()
    end

    n ≥ 0 || throw(ArgumentError("n doit être positif ou nul."))
    return [draw() for _ in 1:n]
end
"""


function ProbaCM(t::Real,gp::GammaProcess, mm::MaintenanceModel)
    # Probability of CM at time t before the first maintenance, given the parameters of the gamma process and the maintenance model.
    return ccdf(Gamma(gp.α * t^gp.β, gp.θ),mm.L)
end

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
    # =1 si β<1 (concave)
    # =2 si β=1 (homogène)
    # =3 si β>1 (convexe)
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


function logprioralpha_at_w(prior, α, β, θ, w)
    if prior isa NonInformative
        return log(pdf(prior, α))
    end
    if !isfinite(w) || w <= 0
        throw(ArgumentError("Le temps w doit être fini et strictement positif."))
    end
    scale = θ * w^β
    return log(pdf(prior, α * scale)) + log(scale)
end

function logcondpostdistbeta(β,priors::Vector{ContinuousUnivariateDistribution},mydf::DataFrame,α,θ,ρ,lbrho; w::Real=1.0)
    beta_prior = priors[2]
    if β <= 0
        return -Inf
    elseif beta_prior isa Beta && β > 1
        return -Inf
    elseif beta_prior isa LocationScale && minimum(beta_prior) >= 1 && β < 1
        return -Inf
    end
    
    s = 0.0
    for i in 1:nrow(mydf)
        eta = deltaeta(mydf, α, β)[i]
        exposure = mydf.deg[i] - exp(mydf.um1[i] * log(1 - ρ)) * (1 - mydf.Δm1[i]) * mydf.degprec[i]
        s += (eta - 1) * log(exposure) - log(θ) * eta - log(gamma(eta))
    end
    s += β*params(priors[1])[1]*log(w)-α*θ*w^(β)/params(priors[1])[2]+log(pdf(beta_prior, β))
    return s
end

function logcondpostdistalpha(α,priors::Vector{ContinuousUnivariateDistribution},mydf::DataFrame,β,θ,ρ,lbrho; w::Real=1.0)
    if α <= 0
        return -Inf
    end

    s = 0.0
    for i in 1:nrow(mydf)
        eta = deltaeta(mydf, α, β)[i]
        exposure = mydf.deg[i] - exp(mydf.um1[i] * log(1 - ρ)) * (1 - mydf.Δm1[i]) * mydf.degprec[i]
        s += (eta - 1) * log(exposure) - log(θ) * eta - log(gamma(eta))
    end
    s += -α*θ*w^(β)/params(priors[1])[2] + (params(priors[1])[1]-1)*log(α) 
    return s
end

function logcondpostdistrho(ρ,priors::Vector{ContinuousUnivariateDistribution},mydf::DataFrame,α,β,θ,lbrho)
    prior_ρ = priors[4]
    if ρ <= lbrho || ρ >= 1
        return -Inf
    else
        return sum((deltaeta(mydf,α,β).-1).*log.(mydf.deg .- exp.(mydf.um1 .* log(1-ρ)) .* (1 .- mydf.Δm1) .* mydf.degprec)) - sum(mydf.deg .- exp.(mydf.um1 .* log(1-ρ)) .* (1 .- mydf.Δm1) .* mydf.degprec)/θ + log(pdf(prior_ρ,ρ))
    end    
end


function mcmc_diagnostics(samples::DataFrame; acceptance_rates=nothing, max_lag::Union{Nothing,Integer}=nothing)
    n = nrow(samples)
    n >= 2 || throw(ArgumentError("Il faut au moins deux itérations pour calculer les diagnostics MCMC."))
    max_lag === nothing || max_lag >= 0 ||
        throw(ArgumentError("max_lag doit être positif ou nul."))
    lag_limit = min(max_lag === nothing ? 1000 : Int(max_lag), n - 1)

    parameter_names = (:α, :β, :θ, :ρ)
    all(name -> name in propertynames(samples), parameter_names) ||
        throw(ArgumentError("La chaîne doit contenir les colonnes α, β, θ et ρ."))

    rates = acceptance_rates === nothing ?
        (α=mean(diff(samples.α) .!= 0), β=mean(diff(samples.β) .!= 0),
         θ=missing, ρ=mean(diff(samples.ρ) .!= 0)) : acceptance_rates
    ess = Dict{Symbol,Union{Missing,Float64}}()
    autocorrelation = DataFrame(lag=collect(0:lag_limit))

    for name in parameter_names
        x = Float64.(samples[!, name])
        all(isfinite, x) ||
            throw(ArgumentError("La chaîne du paramètre $name contient des valeurs non finies."))
        centered = x .- mean(x)
        variance_sum = sum(abs2, centered)

        if variance_sum == 0
            autocorrelation[!, name] = Union{Missing,Float64}[missing for _ in 0:lag_limit]
            ess[name] = missing
            continue
        end

        acf = Vector{Float64}(undef, lag_limit + 1)
        acf[1] = 1.0
        for lag in 1:lag_limit
            acf[lag + 1] = sum(centered[1:(n-lag)] .* centered[(lag+1):n]) / variance_sum
        end
        autocorrelation[!, name] = acf

        positive_pairs_sum = 0.0
        for lag in 1:2:(lag_limit - 1)
            pair_sum = acf[lag + 1] + acf[lag + 2]
            pair_sum > 0 || break
            positive_pairs_sum += pair_sum
        end
        ess[name] = n / (1 + 2 * positive_pairs_sum)
    end

    return (acceptance_rates=rates, ess=ess, autocorrelation=autocorrelation)
end

function mcmc_diagnostics(chains::AbstractVector{<:DataFrame}; acceptance_rates=nothing, max_lag::Union{Nothing,Integer}=nothing)
    n_chains = length(chains)
    n_chains >= 2 || throw(ArgumentError("Il faut au moins deux chaînes MCMC."))
    n = nrow(first(chains))
    n >= 2 || throw(ArgumentError("Chaque chaîne doit contenir au moins deux itérations."))
    all(chain -> nrow(chain) == n, chains) ||
        throw(ArgumentError("Toutes les chaînes doivent avoir la même longueur."))
    max_lag === nothing || max_lag >= 0 ||
        throw(ArgumentError("max_lag doit être positif ou nul."))
    lag_limit = min(max_lag === nothing ? 1000 : Int(max_lag), n - 1)

    parameter_names = (:α, :β, :θ, :ρ)
    all(chain -> all(name -> name in propertynames(chain), parameter_names), chains) ||
        throw(ArgumentError("Chaque chaîne doit contenir les colonnes α, β, θ et ρ."))
    rates = acceptance_rates === nothing ?
        (α=mean([mean(diff(chain.α) .!= 0) for chain in chains]),
         β=mean([mean(diff(chain.β) .!= 0) for chain in chains]),
         θ=missing,
         ρ=mean([mean(diff(chain.ρ) .!= 0) for chain in chains])) : acceptance_rates

    ess = Dict{Symbol,Union{Missing,Float64}}()
    rhat = Dict{Symbol,Union{Missing,Float64}}()
    autocorrelation = DataFrame(lag=collect(0:lag_limit))
    total_iterations = n_chains * n

    for name in parameter_names
        values = [Float64.(chain[!, name]) for chain in chains]
        all(x -> all(isfinite, x), values) ||
            throw(ArgumentError("Une chaîne du paramètre $name contient des valeurs non finies."))
        chain_means = mean.(values)
        chain_variances = [sum(abs2, x .- mean(x)) / (n - 1) for x in values]
        within_variance = mean(chain_variances)
        between_variance = n * sum(abs2, chain_means .- mean(chain_means)) / (n_chains - 1)
        variance_plus = ((n - 1) / n) * within_variance + between_variance / n
        rhat[name] = within_variance == 0 ? missing : sqrt(variance_plus / within_variance)

        if variance_plus == 0
            autocorrelation[!, name] = Union{Missing,Float64}[missing for _ in 0:lag_limit]
            ess[name] = missing
            continue
        end

        rho = Vector{Float64}(undef, lag_limit + 1)
        rho[1] = 1.0
        for lag in 1:lag_limit
            mean_autocovariance = mean([
                sum((x[1:(n-lag)] .- mean(x)) .* (x[(lag+1):n] .- mean(x))) / n
                for x in values
            ])
            rho[lag + 1] = 1 - (within_variance - mean_autocovariance) / variance_plus
        end
        autocorrelation[!, name] = rho

        positive_pairs_sum = 0.0
        for lag in 1:2:(lag_limit - 1)
            pair_sum = rho[lag + 1] + rho[lag + 2]
            pair_sum > 0 || break
            positive_pairs_sum += pair_sum
        end
        ess[name] = min(Float64(total_iterations), total_iterations / (1 + 2 * positive_pairs_sum))
    end

    return (acceptance_rates=rates, ess=ess, rhat=rhat, autocorrelation=autocorrelation)
end


# Faire une fonction qui prend une DataFrame (données) et les loi a priori
# Renvoie un échantillon des paramètres après un algo MCMC (Gibbs + MH)

function algoMCMC(gp::GammaProcess,df::DataFrame, priors::Vector{ContinuousUnivariateDistribution}=[NonInformative(:α), NonInformative(:β), NonInformative(:θ), Uniform()], K::Int=10000,tau2α::Float64=1.0,tau2β::Float64=0.2,tau2ρ::Float64=0.2; w::Real=1.0, x0::Vector{Float64}, return_diagnostics::Bool=false, max_lag::Union{Nothing,Integer}=nothing)
    K > 0 || throw(ArgumentError("K doit être strictement positif."))
    # Choisir des valeurs initiales pour les paramètres (on prend le MLE)
    dα = priors[1]
    dβ = priors[2]
    dθ = priors[3]
    dρ = priors[4]   
    if !(dα isa NonInformative) && (!isfinite(w) || w <= 0)
        throw(ArgumentError("Le temps w doit être fini et strictement positif lorsque le prior de alpha est informatif."))
    end
    mydf = predf(gp::GammaProcess, df::DataFrame)
    lbrho = lowerboundrho(mydf)
    #parinit = MLE(gp,mydf,x0 = [1.0, 1.5, 1.0, (1+lbrho)/2],lower = [1e-2,0.1,0.01,lbrho+0.01],upper=[Inf, Inf, Inf, 1])
    parinit = x0
    estpar = DataFrame((α=zeros(K+1),β=zeros(K+1),θ=zeros(K+1),ρ=zeros(K+1)))
    estpar[1,:]=parinit
    accepted = zeros(Int, 3)
    for k in 2:K+1
        #println("k = $k")
        for j ∈ [3,2,1,4]
            if j==3
                #if dα isa NonInformative
                #    estpar[k,j] = rand(InverseGamma(sum(deltaeta(mydf,estpar[k-1,1],estpar[k-1,2])), sum(mydf.deg .- exp.(mydf.um1 .* log(1-estpar[k-1,4])) .* (1 .- mydf.Δm1) .* mydf.degprec)))
                #else
                alpha_shape, alpha_scale = params(dα)
                theta_shape, theta_scale = dθ isa NonInformative ? (0.0, 0.0) : params(dθ)
                total_shape = sum(deltaeta(mydf, estpar[k-1,1], estpar[k-1,2]))
                total_rate = sum(mydf.deg .- exp.(mydf.um1 .* log(1-estpar[k-1,4])) .* (1 .- mydf.Δm1) .* mydf.degprec)
                p1 = alpha_shape - theta_shape - total_shape
                p3 = 2 * estpar[k-1,1] * w^(estpar[k-1,2]) / alpha_scale
                p2 = 2 * (theta_scale + total_rate)
                #println("p = $p1, a = $p2, b = $p3")
                estpar[k,j] = rand_gig(p1, p2, p3)
                #println(estpar[k,j])
            end
            if j==2
                βstar = rand(Normal(estpar[k-1,2],sqrt(tau2β)))
                logratio = logcondpostdistbeta(βstar,priors,mydf,estpar[k-1,1],estpar[k,3],estpar[k-1,4],lbrho; w=w)-logcondpostdistbeta(estpar[k-1,2],priors,mydf,estpar[k-1,1],estpar[k,3],estpar[k-1,4],lbrho; w=w)
                if logratio > log(rand(Uniform()))
                    estpar[k,2] = βstar
                    accepted[2] += 1
                else
                    estpar[k,2] = estpar[k-1,2]
                end
                    # Cas Non Informatif
                    #βstar = rand(Normal(estpar[k-1,2],sqrt(tau2β)))
                    #println("βstar =  ",βstar)
                    #logratio  = logcondpostdistbeta(βstar,priors,mydf,estpar[k-1,1],estpar[k,3],estpar[k-1,4],lbrho; w=w)-logcondpostdistbeta(estpar[k-1,2],priors,mydf,estpar[k,3],estpar[k-1,1],estpar[k-1,4],lbrho; w=w)
                    #if logratio > log(rand(Uniform()))
                    #    estpar[k,2] = βstar
                    #else
                    #    estpar[k,2] = estpar[k-1,2]
                    #end
            end
            if j==1 
                αstar = rand(Normal(estpar[k-1,1],sqrt(tau2α)))
                logratio  = logcondpostdistalpha(αstar,priors,mydf,estpar[k,2],estpar[k,3],estpar[k-1,4],lbrho; w=w)-logcondpostdistalpha(estpar[k-1,1],priors,mydf,estpar[k,2],estpar[k,3],estpar[k-1,4],lbrho; w=w)
                if logratio > log(rand(Uniform()))
                    estpar[k,1] = αstar
                    accepted[1] += 1
                else
                    estpar[k,1] = estpar[k-1,1]
                end    
            end
            if j==4
                ρstar = rand(Normal(estpar[k-1,4],sqrt(tau2ρ)))
                logratio  = logcondpostdistrho(ρstar,priors,mydf,estpar[k,1],estpar[k,2],estpar[k,3],lbrho)-logcondpostdistrho(estpar[k-1,4],priors,mydf,estpar[k,1],estpar[k,2],estpar[k,3],lbrho)
                if logratio > log(rand(Uniform()))
                    estpar[k,4] = ρstar
                    accepted[3] += 1
                else
                    estpar[k,4] = estpar[k-1,4]
                end
            end
        end
    end
    if return_diagnostics
        acceptance_rates = (α=accepted[1] / K, β=accepted[2] / K, θ=missing, ρ=accepted[3] / K)
        return (samples=estpar, diagnostics=mcmc_diagnostics(estpar[2:end, :]; acceptance_rates, max_lag))
    end
    return estpar
end

