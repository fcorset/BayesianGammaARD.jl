rm(list = ls())

# Cas concaves

LF <- list.files(path = "./Results/Ntrajectoires/Concave/", pattern = ".Rdata")
nb.fic <- length(LF)

tmp <- strsplit(x = LF, split = "_")
tmp <- unlist(x = tmp)
tmp <- matrix(data = tmp, ncol = 8, byrow = TRUE)
HT <- tmp[, 8]
HT <- strsplit(x = HT, split = ".Rdata")
HT <- unlist(x = HT)


label.theta <- c("alpha", "beta", "b", "rho", "rho_w", "p")

library(rlang)

for (i in 1:nb.fic) {
  Fic <- paste("./Results/Ntrajectoires/Concave/", LF[i], sep = "")
  load(file = Fic)
  for (j in 1:6) {
    Fig.File.Name <- paste("./Results/Fig/Concave/",
                           "HISTO_", label.theta[j], "_",
                           tmp[i, 2], "_",
                           tmp[i, 3], "_",
                           tmp[i, 4], "_",
                           tmp[i, 5], "_",
                           tmp[i, 6], "_",
                           tmp[i, 7], "_",
                           HT[i], ".png", sep = "")
    titre <- expr(paste("Histogram of ", !!label.theta[j],
                        " (for ",alpha, "=", !!tmp[i, 2], ", ",
                        beta, "=", !!tmp[i, 3],  ", ",
                        b, "=", !!tmp[i, 4],  ", ",
                        rho, "=", !!tmp[i, 5],  ", ",
                        rho[w], "=", !!tmp[i, 6],  ", ",
                        p, "=", !!tmp[i, 7],  ", ",
                        n, "=", !!HT[i], ")",
                        sep = ""))
    png(Fig.File.Name)
    hist(x = hat.theta[, j], freq = FALSE, nclass = 20, xlab = "", main = titre)
    lines(density(hat.theta[, j]), col = "red", lwd = 2)
    dev.off()
  }
}
