rm(list = ls())

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
MSE.norm  <- matrix(nrow = nb.fic, ncol = 6)
for (i in 1:nb.fic) {
  Fic <- paste("./Results/Ntrajectoires/Concave/", LF[i], sep = "")
  load(file = Fic)
  nb.obs <- nrow(x = hat.theta)
  mat.tmp <- rep(x = tmp[i, ], times = nb.obs)
  mat.tmp <- matrix(data = mat.tmp, nrow = nb.obs, byrow = TRUE)
  mse <- apply(X = (hat.theta - mat.tmp)^2, MARGIN = 2, FUN = mean)
  MSE[i, ] <- mse
  MSE.norm[i, ] <- mse/(tmp[i, ]^2)
}

Tab.MSE.Conc <- data.frame(tmp, HT, MSE, MSE.norm)
colnames(Tab.MSE.Conc) <- c("alpha", "beta", "b", "rho", "rho_w", "p", "HT",
                            "mse.alpha", "mse.beta", "mse.b", "mse.rho",
                            "mse.rho_w", "mse.p", "norm.mse.alpha", "norm.mse.beta",
                            "norm.mse.b", "norm.mse.rho",
                            "norm.mse.rho_w", "norm.mse.p")

# Cas convexes

LF <- list.files(path = "./Results/Ntrajectoires/Convexe/", pattern = ".Rdata")
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
MSE.norm  <- matrix(nrow = nb.fic, ncol = 6)
for (i in 1:nb.fic) {
  Fic <- paste("./Results/Ntrajectoires/Convexe/", LF[i], sep = "")
  load(file = Fic)
  nb.obs <- nrow(x = hat.theta)
  mat.tmp <- rep(x = tmp[i, ], times = nb.obs)
  mat.tmp <- matrix(data = mat.tmp, nrow = nb.obs, byrow = TRUE)
  mse <- apply(X = (hat.theta - mat.tmp)^2, MARGIN = 2, FUN = mean)
  MSE[i, ] <- mse
  MSE.norm[i, ] <- mse/(tmp[i, ]^2)
}

Tab.MSE.Conv <- data.frame(tmp, HT, MSE, MSE.norm)
colnames(Tab.MSE.Conv) <- c("alpha", "beta", "b", "rho", "rho_w", "p", "HT",
                            "mse.alpha", "mse.beta", "mse.b", "mse.rho",
                            "mse.rho_w", "mse.p", "norm.mse.alpha", "norm.mse.beta",
                            "norm.mse.b", "norm.mse.rho",
                            "norm.mse.rho_w", "norm.mse.p")


# Cas homo

LF <- list.files(path = "./Results/Ntrajectoires/Homogene/", pattern = ".Rdata")
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
MSE.norm  <- matrix(nrow = nb.fic, ncol = 6)
for (i in 1:nb.fic) {
  Fic <- paste("./Results/Ntrajectoires/Homogene/", LF[i], sep = "")
  load(file = Fic)
  nb.obs <- nrow(x = hat.theta)
  mat.tmp <- rep(x = tmp[i, ], times = nb.obs)
  mat.tmp <- matrix(data = mat.tmp, nrow = nb.obs, byrow = TRUE)
  mse <- apply(X = (hat.theta - mat.tmp)^2, MARGIN = 2, FUN = mean)
  MSE[i, ] <- mse
  MSE.norm[i, ] <- mse/(tmp[i, ]^2)
}

Tab.MSE.Homo <- data.frame(tmp, HT, MSE, MSE.norm)
colnames(Tab.MSE.Homo) <- c("alpha", "beta", "b", "rho", "rho_w", "p", "HT",
                            "mse.alpha", "mse.beta", "mse.b", "mse.rho",
                            "mse.rho_w", "mse.p", "norm.mse.alpha", "norm.mse.beta",
                            "norm.mse.b", "norm.mse.rho",
                            "norm.mse.rho_w", "norm.mse.p")



# Merging and expprting

Tab.MSE <- rbind(Tab.MSE.Conc, Tab.MSE.Homo, Tab.MSE.Conv)

library(WriteXLS)
do.rounding <- TRUE
nb.dec <- 5
if (do.rounding) {
  Tab.MSE[, 8:19] <- round(x = Tab.MSE[, 8:19], digits = nb.dec)
}
WriteXLS(x = Tab.MSE, ExcelFileName = "MSE.xls")





