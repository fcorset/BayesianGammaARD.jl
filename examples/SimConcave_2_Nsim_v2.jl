# ==============================================================================
# Script de Simulation et d'Estimation (MLE & Bayes multi-priors)
# Optimisé pour l'exécution multi-threadée (192 cœurs) sur Kraken-cpu (GRICAD)
# ==============================================================================
# Tout en haut du fichier, avant "using BayesianGammaARD"
if !("/bettik/PROJECTS/pr-baydeg/fcorset/BayesianGammaARD.jl" in LOAD_PATH)
    push!(LOAD_PATH, "/bettik/PROJECTS/pr-baydeg/fcorset/BayesianGammaARD.jl")
    push!(LOAD_PATH, "/bettik/PROJECTS/pr-baydeg/fcorset/BayesianGammaARD.jl/src")
end

using BayesianGammaARD
using Distributions
using DataFrames
using CSV
using Statistics
using LinearAlgebra 
using Base.Threads
using MCMCDiagnosticTools 

# ------------------------------------------------------------------------------
# CONFIGURATION CRITIQUE POUR KRAKEN-CPU
# ------------------------------------------------------------------------------
# Empêche OpenBLAS/MKL de sur-souscrire les cœurs AMD Genoa en interne.
# Laisse Julia gérer l'intégralité du parallélisme via les 192 threads.
BLAS.set_num_threads(1)

# Vraies valeurs des paramètres
αtrue = 2.0
βtrue = 0.6
θtrue = 2.0
ρtrue = 0.2
rho_label = "rho0.2"

# Définition du modèle générateur
mm = MaintenanceModel(ρ=ρtrue)
gp = GammaProcess(α=αtrue, β=βtrue, θ=θtrue, mm=mm)

HT = 200 # Fenêtre d'observation [0,T]
 n_values = [10, 20, 30, 50, 75, 100, 125, 150, 175, 200]
# test en mode rapide pour débogage
#n_values = [10, 20]

N = 200 # Nombre de simulations globales

# Conteneurs globaux pré-alloués (Un élément par simulation)
# Sécurisé pour le multi-threading car l'index d'écriture 'nsim' est unique par thread.
sorties_sim = Vector{DataFrame}(undef, N)
sorties_MLE = Vector{DataFrame}(undef, N)
sorties_Bayes = Vector{DataFrame}(undef, N)

println("Démarrage des simulations parallèles sur ", Threads.nthreads(), " threads...")

# ------------------------------------------------------------------------------
# BOUCLE PRINCIPALE PARALLÉLISÉE
# ------------------------------------------------------------------------------
Threads.@threads for nsim in 1:N
    # 1. Simulation du processus de dégradation
    y, df = rand(gp, tinsp=1:HT, HT=HT)
    sorties_sim[nsim] = df

    # 2. Initialisation des DataFrames LOCAUX au thread pour collecter les résultats
    local_mle_df = DataFrame(nsim=Int[], n=Int[], α_mle=Float64[], β_mle=Float64[], θ_mle=Float64[], ρ_mle=Float64[], ρlow=Float64[])
    
    local_bayes_df = DataFrame(
        nsim=Int[], n=Int[], prior_type=String[],
        α_mean=Float64[], α_std=Float64[], α_q025=Float64[], α_q975=Float64[], α_ess=Float64[], α_rhat=Float64[], α_acc=Float64[],
        β_mean=Float64[], β_std=Float64[], β_q025=Float64[], β_q975=Float64[], β_ess=Float64[], β_rhat=Float64[], β_acc=Float64[],
        θ_mean=Float64[], θ_std=Float64[], θ_q025=Float64[], θ_q975=Float64[], θ_ess=Float64[], θ_rhat=Float64[], θ_acc=Float64[],
        ρ_mean=Float64[], ρ_std=Float64[], ρ_q025=Float64[], ρ_q975=Float64[], ρ_ess=Float64[], ρ_rhat=Float64[], ρ_acc=Float64[]
    )

    # 3. Boucle interne sur les tailles d'échantillons croissants (n)
    for i in 1:length(n_values)
        n_actuel = n_values[i]
        
        # Utilisation d'une vue "view" pour éviter les duplications de lignes en mémoire RAM
        df_sub = view(df, 1:n_actuel, :)
        mydf = BayesianGammaARD.predf(gp, df_sub)
        ρlow_i = BayesianGammaARD.lowerboundrho(mydf)
        
        # ----------------------------------------------------------------------
        # PARTIE A : MAXIMUM DE VRAISEMBLANCE (MLE)
        # ----------------------------------------------------------------------
        x0_mle = [1.0, 1.5, 1.0, (1 + ρlow_i) / 2]
        res_mle = MLE(gp, mydf, x0_mle, [1e-2, 0.1, 0.01, ρlow_i + 0.01], [Inf, Inf, Inf, 1])
        push!(local_mle_df, (nsim, n_actuel, res_mle..., ρlow_i))

        # ----------------------------------------------------------------------
        # PARTIE B : ESTMATION BAYÉSIENNE MULTI-CHAÎNES
        # ----------------------------------------------------------------------
        w = 1.0
        K = 10000
        # K = 100 # Test rapide pour débogage

        burn_in = 1001
        # burn_in = 10 # Test rapide pour débogage
        n_post_burn = K - burn_in + 1

        # ======================================================================
        # CONFIGURATION DES 3 TYPES DE PRIORS
        # ======================================================================
        
        # --- 1. Good Priors (Correctement centrés) ---
        priormeanθ = θtrue; priorvarθ = 0.5
        a_g = 2 + priormeanθ^2 / priorvarθ; b_g = (a_g - 1) * priormeanθ
        dθ_g = Informative(:θ, a_g, b_g, 1)

        priormeanβ = βtrue; priorvarβ = 0.1
        κ_g = priormeanβ * (1 - priormeanβ) / priorvarβ - 1
        c_g = priormeanβ * κ_g; d_g = (1 - priormeanβ) * κ_g
        dβ_g = Informative(:β, c_g, d_g, 1)

        priormeanEw = θtrue * αtrue * w^βtrue; priorvarEw = 0.5
        f_g = priorvarEw / priormeanEw; e_g = priormeanEw / f_g
        dα_g = Informative(:α, e_g, f_g, 1)

        priormeanρ = 0.2; priorvarρ = 0.05
        kappa_g = priormeanρ * (1 - priormeanρ) / priorvarρ - 1
        g_g = priormeanρ * kappa_g; h_g = (1 - priormeanρ) * kappa_g
        dρ_g = Informative(:ρ, g_g, h_g, 1)
        priors_GP = ContinuousUnivariateDistribution[dα_g, dβ_g, dθ_g, dρ_g]

        # --- 2. Moderately Misspecified Priors (Moderate) ---
        badpriormeanθ = θtrue + 0.5; badpriorvarθ = 0.8
        a_b = 2 + badpriormeanθ^2 / badpriorvarθ; b_b = (a_b - 1) * badpriormeanθ
        dθ_b = Informative(:θ, a_b, b_b, 1)

        badpriormeanβ = βtrue + 0.15; badpriorvarβ = 0.1
        κ_b = badpriormeanβ * (1 - badpriormeanβ) / badpriorvarβ - 1
        c_b = badpriormeanβ * κ_b; d_b = (1 - badpriormeanβ) * κ_b
        dβ_b = Informative(:β, c_b, d_b, 1)

        badpriormeanEw = θtrue * αtrue * w^βtrue + 0.5; badpriorvarEw = 1.0
        f_b = badpriorvarEw / badpriormeanEw; e_b = badpriormeanEw / f_b
        dα_b = Informative(:α, e_b, f_b, 1)

        badpriormeanρ = 0.4; badpriorvarρ = 0.1
        kappa_b = badpriormeanρ * (1 - badpriormeanρ) / badpriorvarρ - 1
        g_b = badpriormeanρ * kappa_b; h_b = (1 - badpriormeanρ) * kappa_b
        dρ_b = Informative(:ρ, g_b, h_b, 1)
        priors_BP = ContinuousUnivariateDistribution[dα_b, dβ_b, dθ_b, dρ_b]

        # --- 3. Strongly Misspecified Priors (Strong) ---
        strongpriormeanθ = θtrue + 2.0; strongpriorvarθ = 0.2
        a_sb = 2 + strongpriormeanθ^2 / strongpriorvarθ; b_sb = (a_sb - 1) * strongpriormeanθ
        dθ_sb = Informative(:θ, a_sb, b_sb, 1)

        strongpriormeanβ = βtrue - 0.3; strongpriorvarβ = 0.02
        κ_sb = strongpriormeanβ * (1 - strongpriormeanβ) / strongpriorvarβ - 1
        c_sb = strongpriormeanβ * κ_sb; d_sb = (1 - strongpriormeanβ) * κ_sb
        dβ_sb = Informative(:β, c_sb, d_sb, 1)

        strongpriormeanEw = θtrue * αtrue * w^βtrue + 2.6; strongpriorvarEw = 0.3
        f_sb = strongpriorvarEw / strongpriormeanEw; e_sb = strongpriormeanEw / f_sb
        dα_sb = Informative(:α, e_sb, f_sb, 1)

        strongpriormeanρ = 0.7; strongpriorvarρ = 0.02
        kappa_sb = strongpriormeanρ * (1 - strongpriormeanρ) / strongpriorvarρ - 1
        g_sb = strongpriormeanρ * kappa_sb; h_sb = (1 - strongpriormeanρ) * kappa_sb
        dρ_sb = Informative(:ρ, g_sb, h_sb, 1)
        priors_SBP = ContinuousUnivariateDistribution[dα_sb, dβ_sb, dθ_sb, dρ_sb]

        # ======================================================================
        # FONCTION INTERNE D'ÉCHANTILLONNAGE ET D'INSERTION
        # ======================================================================
        function executer_et_pousser_mcmc!(priors_list, prior_label)
            mat_α = Matrix{Float64}(undef, n_post_burn, 4)
            mat_β = Matrix{Float64}(undef, n_post_burn, 4)
            mat_θ = Matrix{Float64}(undef, n_post_burn, 4)
            mat_ρ = Matrix{Float64}(undef, n_post_burn, 4)
            
            acc_α_vec = zeros(4); acc_β_vec = zeros(4); acc_ρ_vec = zeros(4)

            # Échantillonnage des 4 chaînes avec dispersion initiale autour du MLE
            for c in 1:4
                x0_chain = [
                    res_mle[1] * rand(Uniform(0.8, 1.2)),
                    res_mle[2] * rand(Uniform(0.8, 1.2)),
                    res_mle[3] * rand(Uniform(0.8, 1.2)),
                    max(ρlow_i + 0.01, min(0.99, res_mle[4] * rand(Uniform(0.8, 1.2))))
                ]
                
                # Exécution MCMC via votre fonction exacte
                res_MCMC = algoMCMC(gp, mydf, priors_list, K, 1.0, 0.2, 0.2; w=w, x0=x0_chain, return_diagnostics=false)
                
                # Remplissage des matrices post-burn-in (via vos symboles :α, :β, :θ, :ρ)
                mat_α[:, c] = res_MCMC[burn_in:end, :α]
                mat_β[:, c] = res_MCMC[burn_in:end, :β]
                mat_θ[:, c] = res_MCMC[burn_in:end, :θ]
                mat_ρ[:, c] = res_MCMC[burn_in:end, :ρ]
                
                acc_α_vec[c] = mean(diff(res_MCMC[burn_in:end, :α]) .!= 0)
                acc_β_vec[c] = mean(diff(res_MCMC[burn_in:end, :β]) .!= 0)
                acc_ρ_vec[c] = mean(diff(res_MCMC[burn_in:end, :ρ]) .!= 0)
            end

            # Diagnostics joints R-hat / ESS (MCMCDiagnosticTools)
            samples_3D = Array{Float64, 3}(undef, n_post_burn, 4, 4)
            samples_3D[:, :, 1] = mat_α; samples_3D[:, :, 2] = mat_β
            samples_3D[:, :, 3] = mat_θ; samples_3D[:, :, 4] = mat_ρ
            diag_res = ess_rhat(samples_3D)

            # Vrais quantiles empiriques exacts sur les 4 chaînes fusionnées
            q_α = quantile(vec(mat_α), [0.025, 0.975])
            q_β = quantile(vec(mat_β), [0.025, 0.975])
            q_θ = quantile(vec(mat_θ), [0.025, 0.975])
            q_ρ = quantile(vec(mat_ρ), [0.025, 0.975])

            # Push dans la structure locale
            push!(local_bayes_df, (
                nsim, n_actuel, prior_label,
                mean(mat_α), std(mat_α), q_α[1], q_α[2], diag_res.ess[1], diag_res.rhat[1], mean(acc_α_vec),
                mean(mat_β), std(mat_β), q_β[1], q_β[2], diag_res.ess[2], diag_res.rhat[2], mean(acc_β_vec),
                mean(mat_θ), std(mat_θ), q_θ[1], q_θ[2], diag_res.ess[3], diag_res.rhat[3], 0.0, # θ est conjugué GIG (acc = 100%)
                mean(mat_ρ), std(mat_ρ), q_ρ[1], q_ρ[2], diag_res.ess[4], diag_res.rhat[4], mean(acc_ρ_vec)
            ))
        end
    # Exécution séquentielle contrôlée des 3 scénarios de priors
    executer_et_pousser_mcmc!(priors_GP, "good")
    executer_et_pousser_mcmc!(priors_BP, "moderate")
    executer_et_pousser_mcmc!(priors_SBP, "strong")
    end
# Sauvegarde des DataFrames locaux dans les conteneurs partagés
sorties_MLE[nsim] = local_mle_df
sorties_Bayes[nsim] = local_bayes_df
end
println("Simulations et MCMC terminées. Fusion et post-traitement global des indicateurs...")

#------------------------------------------------------------------------------
#          POST-TRAITEMENT ET CALCUL DES INDICATEURS D'ARTICLE
#------------------------------------------------------------------------------

df_final_mle = reduce(vcat, sorties_MLE)
df_final_bayes = reduce(vcat, sorties_Bayes)

# Fonction d'agrégation statistique sur l'ensemble des N=200 simulations

function calculer_metrics_performance(df_bayes, param_name, true_val)
mean_col = Symbol(param_name, "_mean")
q025_col = Symbol(param_name, "_q025")
q975_col = Symbol(param_name, "_q975")
df_bayes[!, :error] = df_bayes[!, mean_col] .- true_val
df_bayes[!, :sq_error] = df_bayes[!, :error].^2
df_bayes[!, :covered] = (df_bayes[!, q025_col] .<= true_val) .& (true_val .<= df_bayes[!, q975_col])
df_bayes[!, :ic_length] = df_bayes[!, q975_col] .- df_bayes[!, q025_col]
synthese = combine(groupby(df_bayes, [:n, :prior_type]),
:error => mean => Symbol("Bias_", param_name),
:sq_error => (x -> sqrt(mean(x))) => Symbol("RMSE_", param_name),
:covered => mean => Symbol("Coverage_", param_name),
:ic_length => mean => Symbol("AvgLength_", param_name)
)
return synthese
end

# Application aux 4 paramètres de dégradation/maintenance

df_perf_α = calculer_metrics_performance(df_final_bayes, "α", αtrue)
df_perf_β = calculer_metrics_performance(df_final_bayes, "β", βtrue)
df_perf_θ = calculer_metrics_performance(df_final_bayes, "θ", θtrue)
df_perf_ρ = calculer_metrics_performance(df_final_bayes, "ρ", ρtrue)

# Fusion générale de la table récapitulative

df_performance_globale = innerjoin(df_perf_α, df_perf_β, on=[:n, :prior_type])
df_performance_globale = innerjoin(df_performance_globale, df_perf_θ, on=[:n, :prior_type])
df_performance_globale = innerjoin(df_performance_globale, df_perf_ρ, on=[:n, :prior_type])
sort!(df_performance_globale, [:prior_type, :n])

#------------------------------------------------------------------------------
#  EXPORTATION SUR L'ESPACE NVMe DU MÉSOCENTRE (/bettik)
#------------------------------------------------------------------------------
dir_out = "/bettik/PROJECTS/pr-baydeg/fcorset/BayesianGammaARD.jl/examples/Results/Nsim/Concave"
if !isdir(dir_out) mkdir(dir_out) end
CSV.write(joinpath(dir_out, "mle_raw_results.csv"), df_final_mle)
CSV.write(joinpath(dir_out, "bayes_raw_results.csv"), df_final_bayes)
CSV.write(joinpath(dir_out, "article_performance_metrics_$(rho_label).csv"), df_performance_globale)
println("Succès ! Vos fichiers LaTeX/CSV de révision se trouvent dans : ", dir_out)