using BayesianGammaARD
using Distributions
using DataFrames
using CSV
using Statistics
using Plots

# Vraies valeurs des paramètres
αtrue = 2
βtrue = 0.7
θtrue = 2
ρtrue = 0.7
rho_label = "rho0.7"

# Définition du modèle
mm = MaintenanceModel(ρ=ρtrue)
gp = GammaProcess(α=αtrue,β=βtrue,θ=θtrue,mm=mm)

# ... (simulation) ...

HT = 200 # Fenêtre d'observation [0,T]

# Simulation du processus de dégradation
y, df = rand(gp,tinsp=1:HT,HT=HT)


# On génère 9 jeux de données (n croissant) avec nom de variable explicite
df10_rho0_7  = df[1:10,:]
df25_rho0_7  = df[1:25,:]
df50_rho0_7  = df[1:50,:]
df75_rho0_7  = df[1:75,:]
df100_rho0_7 = df[1:100,:]
df125_rho0_7 = df[1:125,:]
df150_rho0_7 = df[1:150,:]
df175_rho0_7 = df[1:175,:]
df200_rho0_7 = df[1:200,:]

# On récupère les données depuis des fichiers CSV
df10_rho0_7 = CSV.read("examples/Results/Concave/sim_data_full_$rho_label.csv", DataFrame)[1:10,:]
df25_rho0_7 = CSV.read("examples/Results/Concave/sim_data_full_$rho_label.csv", DataFrame)[1:25,:]
df50_rho0_7 = CSV.read("examples/Results/Concave/sim_data_full_$rho_label.csv", DataFrame)[1:50,:]
df75_rho0_7 = CSV.read("examples/Results/Concave/sim_data_full_$rho_label.csv", DataFrame)[1:75,:]
df100_rho0_7 = CSV.read("examples/Results/Concave/sim_data_full_$rho_label.csv", DataFrame)[1:100,:]
df125_rho0_7 = CSV.read("examples/Results/Concave/sim_data_full_$rho_label.csv", DataFrame)[1:125,:]
df150_rho0_7 = CSV.read("examples/Results/Concave/sim_data_full_$rho_label.csv", DataFrame)[1:150,:]
df175_rho0_7 = CSV.read("examples/Results/Concave/sim_data_full_$rho_label.csv", DataFrame)[1:175,:]
df200_rho0_7 = CSV.read("examples/Results/Concave/sim_data_full_$rho_label.csv", DataFrame)[1:200,:]    

# Plot de la dégradation simulée
plot(df.tinsp, df.deg, label="Simulated degradation", xlabel="Time", ylabel="Degradation level", title="Simulated degradation process (ρ=0.7)")
hline!([3], label="L")
hline!([6], label="M")
savefig("examples/Results/Concave/simulated_degradation_rho0_7.png")


ρlow = Vector{Float64}(undef,9)
EstMLE_rho0_7 = Vector{Vector{Float64}}(undef,9)

res_NI_all_rho0_7 = Vector{DataFrame}(undef,9)
res_GP_all_rho0_7 = Vector{DataFrame}(undef,9)

for i in 1:length(ρlow)
    dfs = [df10_rho0_7, df25_rho0_7, df50_rho0_7, df75_rho0_7, df100_rho0_7,
           df125_rho0_7, df150_rho0_7, df175_rho0_7, df200_rho0_7]
    mydf = BayesianGammaARD.predf(gp, dfs[i])
    ρlow[i] = BayesianGammaARD.lowerboundrho(mydf)
    x0 = [1.0, 0.5, 1.5, (1+ρlow[i])/2]

    EstMLE_rho0_7[i] = MLE(gp, mydf, x0, [1e-2,0.1,0.01,ρlow[i]+0.01], [Inf, Inf, Inf, 1])
    # ... (bayesian estimation) ...
    priors_NI = [
        NonInformative(:α),
        NonInformative(:β),
        NonInformative(:θ),
        (1-ρlow[i])*Uniform() + ρlow[i]
    ]
    res_NI = algoMCMC(gp, mydf, priors_NI, 10000, 1.0, 0.5, 0.2)
    res_NI_all_rho0_7[i] = res_NI[1001:end,:]

    # "good priors" informatifs
    priormeanθ = θtrue
    priorvarθ = 0.5
    a = 2 + priormeanθ^2 / priorvarθ
    b = (a-1) * priormeanθ
    dθ = Informative(:θ, a, b, 1)

    # Bon prior pour β : Beta(c,d) avec c et d choisis pour que la moyenne soit égale à βtrue et la variance à 0.2
    priormeanβ = βtrue
    priorvarβ = 0.1
    c = priorvarβ /(priormeanβ-priormeanβ^2 - priorvarβ/priormeanβ)
    d = c * (1/priormeanβ - 1)
    dβ = Informative(:β, c, d, 1) # le troisième pramètre est égal à 1 pour loi beta

    w = 1
    priormeanEw = θtrue * αtrue * w^βtrue
    priorvarEw = 0.2
    f = priorvarEw / priormeanEw
    e = priormeanEw / f
    dα = Informative(:α, e, f, 2)

    priormeanρ = 0.7
    priorvarρ = 0.02
    if priormeanρ <= ρlow[i]
        priormeanρ = 0.75
        priorvarρ = 0.01  # Variance réduite
        println("Warning: priormeanρ ($priormeanρ) is not greater than ρlow ($(ρlow[i])). $priormeanρ changed.")
    end
    g = (ρlow[i] - priormeanρ) * (ρlow[i] - ρlow[i]*priormeanρ - priormeanρ + priormeanρ^2 + priorvarρ) / (priorvarρ*(1-ρlow[i]))
    h = -(1-priormeanρ) * (ρlow[i] - ρlow[i]*priormeanρ - priormeanρ + priormeanρ^2 + priorvarρ) / (priorvarρ*(1-ρlow[i]))

    if g > 0 && h > 0
        dρ = (1-ρlow[i])*Informative(:ρ,g,h,1)+ρlow[i]
    else
        println("Warning: Invalid parameters for Beta distribution.")
        println(g)
        println(h)
        println(i)
        println("ρlow[i] = $(ρlow[i]), priormeanρ = $(priormeanρ), priorvarρ = $(priorvarρ)")
        priormeanρ = ρlow[i] + 0.1
        priorvarρ = 0.01  # Variance réduite
        g = (ρlow[i]-priormeanρ)*(ρlow[i]-ρlow[i]*priormeanρ-priormeanρ+priormeanρ^2+priorvarρ)/(priorvarρ*(1-ρlow[i]))
        h=-(1-priormeanρ)*(ρlow[i]-ρlow[i]*priormeanρ-priormeanρ+priormeanρ^2+priorvarρ)/(priorvarρ*(1-ρlow[i]))    
        println("New priors for ρ: mean = $priormeanρ, variance = $priorvarρ")
        println("New parameters for Beta distribution: g = $g, h = $h")
    end


    dρ = (1-ρlow[i]) * Informative(:ρ, g, h, 1) + ρlow[i]

    priors_GP = [dα, dβ, dθ, dρ]
    res_GP = algoMCMC(gp, mydf, priors_GP, 10000, 1.0, 0.5, 0.2)
    res_GP_all_rho0_7[i] = res_GP[1001:end,:]
end

# Sauvegarde (fichiers + noms de dataframes) avec indication rho 0.7
CSV.write("examples/Results/Concave/sim_data_full_$rho_label.csv", df200_rho0_7)

n_values = [10,25,50,75,100,125,150,175,200]
for i in 1:length(res_NI_all_rho0_7)
    CSV.write("examples/Results/Concave/bayes_posterior_non_informative_n$(n_values[i])_$rho_label.csv", res_NI_all_rho0_7[i])
    CSV.write("examples/Results/Concave/bayes_posterior_good_priors_n$(n_values[i])_$rho_label.csv", res_GP_all_rho0_7[i])
end

for i in 1:length(EstMLE_rho0_7)
    push!(mle_df_rho_0_7, (n_values[i], EstMLE_rho0_7[i]...))
end
mle_df_rho_0_7[!, :ρlow] = ρlow
CSV.write("examples/Results/Concave/mle_results_$rho_label.csv", mle_df_rho_0_7)





# Graphiques pour le cas ρ=0.7 (cas convexe et non informatif)  

for i in 1:length(ρlow)
    # Graphique α
    p_α = histogram(res_NI_all[i][:,1],normalize=:pdf,
        label="posterior distribution of alpha",
        xlabel="alpha",
        ylabel="density",
        title="Posterior distribution of parameter α (n=$(i*10), Convex, Non-informative prior)",
        legend=:topright,
        xlims=(0,2),
        ylims=(0,3))
    vline!([gp.α],label="true value of alpha")
    vline!([EstMLE[i][1]],label="MLE of alpha")
    vline!([mean(res_NI_all[i][:,1])],label="Bayesian Estimator of alpha")
    savefig(p_α, "examples/Results/Convex/posterior_alpha_non_informative_n$(n_values[i])_$rho_label.png")
    
    # Graphique ρ
    p_ρ = histogram(res_NI_all[i][:,4],normalize=:pdf,
        label="posterior distribution of rho",
        xlabel="rho",
        ylabel="density",
        title="Posterior distribution of parameter ρ (n=$(n_values[i]), Convex, Non-informative prior)",
        xlims=(ρlow[i],1),
        ylims=(0,3))
    vline!([gp.mm.ρ], label="true value of rho")
    vline!([EstMLE[i][4]],label="MLE of rho")
    vline!([mean(res_NI_all[i][:,4])],label="Bayesian Estimator of rho")
    savefig(p_ρ, "examples/Results/Convex/posterior_rho_non_informative_n$(n_values[i])_$rho_label.png")
    
    # Graphique β
    p_β = histogram(res_NI_all[i][:,2],normalize=:pdf,
        label="posterior distribution of beta",
    mle_df_rho_0_7 = DataFrame(n = Int[], α = Float64[], β = Float64[], θ = Float64[], ρ = Float64[])
    xlabel="beta",
        ylabel="density",
        title="Posterior distribution of parameter β (n=$(i*10), Convex, Non-informative prior)",
        xlims=(1,2.5),
        ylims=(0,3))
    vline!([gp.β], label="true value of beta")
    vline!([EstMLE[i][2]],label="MLE of beta")
    vline!([mean(res_NI_all[i][:,2])],label="Bayesian Estimator of beta")
    savefig(p_β, "examples/Results/Convex/posterior_beta_non_informative_n$(n_values[i])_$rho_label.png")

    # Graphique θ
    p_θ = histogram(res_NI_all[i][:,3],normalize=:pdf,
        label="posterior distribution of theta",
        xlabel="theta",
        ylabel="density",
        title="Posterior distribution of parameter θ (n=$(i*10), Convex, Non-informative prior)",
        xlims=(0,5),
        ylims=(0,3))
    vline!([gp.θ], label="true value of theta")
    vline!([EstMLE[i][3]],label="MLE of theta")
    vline!([mean(res_NI_all[i][:,3])],label="Bayesian Estimator of theta")
    savefig(p_θ, "examples/Results/Convex/posterior_theta_non_informative_n$(n_values[i])_$rho_label.png")      
end

for i in 1:length(ρlow)
        # Graphique α
        p_α = histogram(res_GP_all[i][:,1],normalize=:pdf,
            label="posterior distribution of alpha",
            xlabel="alpha",
            ylabel="density",
            title="Posterior distribution of parameter α (n=$(n_values[i]), Convex, Good priors)",
            legend=:topright,
            xlims=(0,2),
            ylims=(0,3))
        vline!([gp.α],label="true value of alpha")
        vline!([EstMLE[i][1]],label="MLE of alpha")
        vline!([mean(res_GP_all[i][:,1])],label="Bayesian Estimator of alpha")
        savefig(p_α, "examples/Results/Convex/posterior_alpha_good_priors_n$(n_values[i])_$rho_label.png")
        
        # Graphique ρ
        p_ρ = histogram(res_GP_all[i][:,4],normalize=:pdf,
            label="posterior distribution of rho",
            xlabel="rho",
            ylabel="density",
            title="Posterior distribution of parameter ρ (n=$(n_values[i]), Convex, Good priors)",
            xlims=(ρlow[i],1),
            ylims=(0,3))
        vline!([gp.mm.ρ], label="true value of rho")
        vline!([EstMLE[i][4]],label="MLE of rho")
        vline!([mean(res_GP_all[i][:,4])],label="Bayesian Estimator of rho")
        savefig(p_ρ, "examples/Results/Convex/posterior_rho_good_priors_n$(n_values[i])_$rho_label.png")
        
        # Graphique β
        p_β = histogram(res_GP_all[i][:,2],normalize=:pdf,
            label="posterior distribution of beta",
            xlabel="beta",
            ylabel="density",
            title="Posterior distribution of parameter β (n=$(n_values[i]), Convex, Good priors)",
            xlims=(1,2.5),
            ylims=(0,3))
        vline!([gp.β], label="true value of beta")
        vline!([EstMLE[i][2]],label="MLE of beta")
        vline!([mean(res_GP_all[i][:,2])],label="Bayesian Estimator of beta")
        savefig(p_β, "examples/Results/Convex/posterior_beta_good_priors_n$(n_values[i])_$rho_label.png")

        # Graphique θ
        p_θ = histogram(res_GP_all[i][:,3],normalize=:pdf,
            label="posterior distribution of theta",
            xlabel="theta",
            ylabel="density",
            title="Posterior distribution of parameter θ (n=$(n_values[i]), Convex, Good priors)",
            xlims=(0,5),
            ylims=(0,3))
        vline!([gp.θ], label="true value of theta")
        vline!([EstMLE[i][3]],label="MLE of theta")
        vline!([mean(res_GP_all[i][:,3])],label="Bayesian Estimator of theta")
        savefig(p_θ, "examples/Results/Convex/posterior_theta_good_priors_n$(n_values[i])_$rho_label.png")
end








for i in 1:length(ρlow)
    dfs = [df10, df25, df50, df75, df100, df125, df150, df175, df200]
    mydf = BayesianGammaARD.predf(gp, dfs[i])
    ρlow[i] = BayesianGammaARD.lowerboundrho(mydf)
    x0 = [1.0, 1.5, 1.0, (1+ρlow[i])/2]

    EstMLE[i] = MLE(gp, mydf, x0, [1e-2,0.1,0.01,ρlow[i]+0.01], [Inf, Inf, Inf, 1])
    # PARTIE BAYESIENNE
    # CAS Non informatif
    priors = [NonInformative(:α), NonInformative(:β), NonInformative(:θ), (1-ρlow[i])*Uniform()+ρlow[i]]
    res_NI = algoMCMC(gp, mydf, priors, 10000, 1.0, 0.5, 0.2)
    res_NI_all[i] = res_NI[1001:end,:] # On enlève les 1000 premières itérations pour éviter l'effet de démarrage

    # Cas Informatif (avec des priors centrés sur les vraies valeurs des paramètres) appelé GP (good priors)
    
    # Bon prior pour θ : Gamma(a,b) avec a et b choisis pour que la moyenne soit égale à θtrue et la variance à 0.5
    priormeanθ = 1.0
    priorvarθ = 0.5
    a = 2+ priormeanθ^2 / priorvarθ
    b = (a-1) * priormeanθ
    dθ = Informative(:θ,a,b,1)

    # Bon prior pour β : Gamma(c,c) avec c choisi pour que la moyenne soit égale à βtrue et la variance à 0.5, et avec une contrainte de β > 1
    priormeanβ = 1.5
    priorvarβ = 0.5

    d = priorvarβ/(priormeanβ-1)
    c = (priormeanβ-1)/d

    dβ = Informative(:β,c,d,3) # le troisième argument (3) indique que c'est un prior pour β >1.

    # Bon prior pour α 

    w = 1 # time given by expert à mettre à jour dans la fonction postalpha....
    priormeanEw = θtrue*αtrue*w^βtrue # Degradation level at time w given by expert, ici la vraie valeur !
    priorvarEw = 0.2
    f = priorvarEw/priormeanEw
    e = priormeanEw/f
    dα = Informative(:α,e,f,2) # Le deuxième paramètre sera mis à jour dans le calcul de la post de alpha

    # Bon prior pour ρ : Loi Beta (g,h)
    priormeanρ = 0.7
    priorvarρ = 0.02

    if priormeanρ <= ρlow[i]
        println("Warning: priormeanρ ($priormeanρ) is not greater than ρlow ($(ρlow[i])). $priormeanρ changed.")
        priormeanρ = (ρlow[i] + 1) / 2
    end

    varpriorρ = 0.02

    g = (ρlow[i]-priormeanρ)*(ρlow[i]-ρlow[i]*priormeanρ-priormeanρ+priormeanρ^2+varpriorρ)/(varpriorρ*(1-ρlow[i]))
    h=-(1-priormeanρ)*(ρlow[i]-ρlow[i]*priormeanρ-priormeanρ+priormeanρ^2+varpriorρ)/(varpriorρ*(1-ρlow[i]))

    # Vérifier si g>0 et h>0 pour que ce soit une loi Beta valide
    if g > 0 && h > 0
        dρ = (1-ρlow[i])*Informative(:ρ,g,h,1)+ρlow[i]
    else
        println("Warning: Invalid parameters for Beta distribution.")
        varpriorρ = 0.01  # Variance réduite
        g = (ρlow[i]-priormeanρ)*(ρlow[i]-ρlow[i]*priormeanρ-priormeanρ+priormeanρ^2+varpriorρ)/(varpriorρ*(1-ρlow[i]))
        h=-(1-priormeanρ)*(ρlow[i]-ρlow[i]*priormeanρ-priormeanρ+priormeanρ^2+varpriorρ)/(varpriorρ*(1-ρlow[i]))    
    end

    dρ = (1-ρlow[i])*Informative(:ρ,g,h,1)+ρlow[i]


    priors_GP = [dα, dβ, dθ, dρ]

    res_GP = algoMCMC(gp, mydf, priors_GP, 10000, 1.0, 0.5, 0.2)
    res_GP_all[i] = res_GP[1001:end,:] # On enlève les 1000 premières itérations pour éviter l'effet de démarrage    

end



# Sauvegarde de la dataframe complète en CSV pour le cas ρ=0.7 (cas convexe et non informatif)
CSV.write("examples/Results/Convex/sim_data_full_0.7.csv", df200)

# Sauvegarde des résultats bayésiens (postérieurs) pour chaque jeu de données et pour le cas ρ=0.7 (cas convexe et non informatif)
n_values = [10, 25, 50, 75, 100, 125, 150, 175, 200]
for i in 1:length(res_NI_all)
    CSV.write("examples/Results/Convex/bayes_posterior_non_informative_n$(n_values[i]).csv", res_NI_all[i])
    CSV.write("examples/Results/Convex/bayes_posterior_good_priors_n$(n_values[i]).csv", res_GP_all[i])
end 





# Analyse des résultats cas convexe et non informatif et ρ=0.7 

# Mettre la loi a priori sur le graphique !
# Violin plot ?


###########################################################################################














mydf = BayesianGammaARD.predf(gp,df10)

plot(mydf.tinsp,mydf.deg)
hline!([3],label="L")
hline!([6],label="M")

ρlow=BayesianGammaARD.lowerboundrho(mydf)

# Initialisation du vecteur de paramètre pour l'optimisation de la fonction log-vraisemblance
x0 = [1.0, 1.5, 1.0, (1+ρlow)/2]

# Calcul du MLE
est10 = MLE(gp,mydf,x0,[1e-2,0.1,0.01,ρlow+0.01],[Inf, Inf, Inf, 1])

# PARTIE BAYESIENNE
# CAS Non informatif

priors = [NonInformative(:α), NonInformative(:β), NonInformative(:θ), (1-ρlow)*Uniform()+ρlow]

res = algoMCMC(gp,df,priors,10000,1.0,0.5,0.2)
res = res[1001:end,:] # On enlève les 1000 premières itérations pour éviter l'effet de démarrage

# Analyse des résultats Prior Non info 

res10_NonInfo = res

histogram(res10_NonInfo[:,1],normalize=:pdf,
    label="posterior distribution of alpha",
    xlabel="alpha",
    ylabel="density",
    title="Posterior distribution of parameter α (n=10)",
    legend=:topright,
    xlims=(0,2),
    ylims=(0,3))
vline!([gp.α],label="true value of alpha")
vline!([est[1]],label="MLE of alpha")
vline!([mean(res10_NonInfo[:,1])],label="Bayesian Estimator of alpha")

histogram(res10_NonInfo[:,2],normalize=:pdf,
    label="posterior distribution of beta",
    xlabel="beta",
    ylabel="density",
        title="Posterior distribution of parameter β (n=10)",
    xlims=(1,2.5),
    ylims=(0,3))
vline!([gp.β], label="true value of beta")
vline!([est[2]],label="MLE of beta")
vline!([mean(res10_NonInfo[:,2])],label="Bayesian Estimator of beta")

histogram(res10_NonInfo[:,3],normalize=:pdf,
    label="posterior distribution of theta",
    xlabel="theta",
    ylabel="density",
    title="Posterior distribution of parameter θ (n=10)",
    xlims=(0,5),
    ylims=(0,3))
vline!([gp.θ], label="true value of theta")
vline!([est[3]],label="MLE of theta")
vline!([mean(res10_NonInfo[:,3])],label="Bayesian Estimator of theta")

histogram(res10_NonInfo[:,4],normalize=:pdf,
    label="posterior distribution of rho",
    xlabel="rho",
    ylabel="density",
    title="Posterior distribution of parameter ρ (n=10)",
    xlims=(ρlow,1),
    ylims=(0,3))
vline!([gp.mm.ρ], label="true value of alpha")
vline!([est[4]],label="MLE of rho")
vline!([mean(res10_NonInfo[:,4])],label="Bayesian Estimator of rho")
# histogram! permet de superposer les histogrammes

# --- Sauvegarde des dataframes dans des fichiers CSV ---
CSV.write("Results/Convex/sim_data_full.csv", df)
for i in 1:10
    subset_df = eval(Meta.parse("df" * string(i*10)))
    CSV.write("Results/Convex/sim_data_$(i*10).csv", subset_df)
end

# Sauvegarde des résultats bayésiens (postérieurs) pour chaque jeu de données
for i in 1:length(res_all)
    CSV.write("Results/Convex/bayes_posterior_n$(i*10).csv", res_all[i])
end

# Sauvegarde des MLE pour chaque jeu de données
mle_df = DataFrame(n = Int[], α = Float64[], β = Float64[], θ = Float64[], ρ = Float64[])
for i in 1:length(EstMLE)
    push!(mle_df, (i*10, EstMLE[i]...))
    CSV.write("Results/Convex/mle_n$(i*10).csv", DataFrame(α = EstMLE[i][1], β = EstMLE[i][2], θ = EstMLE[i][3], ρ = EstMLE[i][4]))
end
CSV.write("Results/Convex/mle_results.csv", mle_df)


