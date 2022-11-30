TabParam <- c(2, 0.8, 2, 0.5, 0.5, 0.5, 100,
2, 0.8, 2, 0.8, 0.1, 0.9, 100,
2, 0.8, 2, 0.5, 0.5, 0.5, 500,
2, 0.8, 2, 0.8, 0.1, 0.9, 500,
2, 0.8, 2, 0.5, 0.5, 0.5, 50,
2, 0.8, 2, 0.8, 0.1, 0.9, 50,
2, 0.8, 2, 0.5, 0.5, 0.9, 100,
2, 0.8, 2, 0.8, 0.5, 0.5, 100,
2, 0.8, 2, 0.5, 0.5, 0.9, 500,
2, 0.8, 2, 0.8, 0.5, 0.5, 500,
2, 0.8, 2, 0.5, 0.5, 0.9, 50,
2, 0.8, 2, 0.8, 0.5, 0.5, 50,
2, 0.8, 2, 0.8, 0.1, 0.5, 100,
2, 0.8, 2, 0.8, 0.5, 0.9, 100,
2, 0.8, 2, 0.8, 0.1, 0.5, 500,
2, 0.8, 2, 0.8, 0.5, 0.9, 500,
2, 0.8, 2, 0.8, 0.1, 0.5, 50,
2, 0.8, 2, 0.8, 0.5, 0.9, 50,
2, 0.8, 2, 0.8, 0.1, 0.7, 100,
2, 0.8, 2, 0.8, 0.7, 0.7, 100,
2, 0.8, 2, 0.8, 0.1, 0.7, 500,
2, 0.8, 2, 0.8, 0.7, 0.7, 500,
2, 0.8, 2, 0.8, 0.1, 0.7, 50,
2, 0.8, 2, 0.8, 0.7, 0.7, 50)

TabParam <- matrix(data = TabParam, ncol = 7, byrow = TRUE)
NbCas <- nrow(x = TabParam)

for (w in 1:NbCas) {
  alpha <- TabParam[w, 1] # paramètre de forme de Gamma a = alpha (t)^beta
  beta <- TabParam[w, 2] # paramètre de forme  de Gamma
  b <- TabParam[w, 3]   # paramètre d'échelle du Gamma
  rho <- TabParam[w, 4] # parametre ARDinf pour les maintenances efficaces avec proba p
  rho_w <- TabParam[w, 5] # parametre ARDinf pour les maintenances néfastes avec proba 1-p
  p <- TabParam[w, 6] # proba que la maintenance préventive soit efficace
  HT <- TabParam[w, 7] # fenêtre d'observation du processus
  source("main (concave).R")
}






