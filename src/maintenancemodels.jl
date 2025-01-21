
mutable struct MaintenanceModel
    L::Float64 # seuil L à partir duquel on déclenche une Maintenance Préventive (ARD)
    U::Float64 # seuil U à partir duquel on déclenche une Maintenance Corrective (AGAN)
    ρ::Float64 # paramètre de l'ARD
end

MaintenanceModel(; L=3.0, U=6.0, ρ=0.5) = MaintenanceModel(L,U,ρ)



