# Cas concaves

LF <- list.files(path = "./Results/Ntrajectoires/Concave/", pattern = ".Rdata")
nb.fic <- length(LF)

tmp <- strsplit(x = LF, split = "_")
tmp <- unlist(x = tmp)
tmp <- matrix(data = tmp, ncol = 8, byrow = TRUE)
tmp <- tmp[, -1]
HT <- tmp[, 7]
HT <- strsplit(x = HT, split = ".Rdata")
HT <- unlist(x = HT)
tmp <- tmp[, -7]
# tmp <- cbind(tmp, HT)
tmp <- apply(X = tmp, MARGIN = 2, FUN = as.numeric)
HT <- as.numeric(HT)

MSE <- matrix(nrow = nb.fic, ncol = 6)
for (i in 1:nb.fic) {
  Fic <- paste("./Results/Ntrajectoires/Concave/", LF[i], sep = "")
  load(file = Fic)
  nb.obs <- nrow(x = hat.theta)
  mat.tmp <- rep(x = tmp[i, ], times = nb.obs)
  mat.tmp <- matrix(data = mat.tmp, nrow = nb.obs, byrow = TRUE)
  mse <- apply(X = (hat.theta - mat.tmp)^2, MARGIN = 2, FUN = mean)
  MSE[i, ] <- mse
}

Tab.MSE.Conc <- data.frame(tmp, HT, MSE)
colnames(Tab.MSE.Conc) <- c("alpha", "beta", "b", "rho", "rho_w", "p", "HT",
                            "mse.alpha", "mse.beta", "mse.b", "mse.rho",
                            "mse.rho_w", "mse.p")




