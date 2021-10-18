alpha <- rep(x = 1, times = 9)
beta <- c(1, 1, 1, 0.5, 0.5, 0.5, 1.2, 1.2, 1.2)
b <- rep(x = 1, times = 9)
rho <- c(0.2, 0.5, 0.8, 0.2, 0.5, 0.8, 0.2, 0.5, 0.8)
TabCas <- cbind(alpha, beta, b, rho)

ListFic <- list.files()
idx <- grep(pattern = "Rd", x = ListFic)
ListFic <- ListFic[idx]

NbFic <- length(x = ListFic)

pdf("Output.plot.pdf")

for (i in 1:NbFic) {
  Fic <- ListFic[i]
  load(file = Fic)
  # aux <- vparms
  # vparms <- apply(X = aux, MARGIN = 2, scale)
  par(mfrow = c(2,2))
  hist(x = vparms[,1], probability = TRUE, xlab = "alpha", main = "")
  lines(density(vparms[,1]),lwd = 2, col = "red")
  hist(x = vparms[,2], probability = TRUE, xlab = "beta", main = "")
  lines(density(vparms[,2]),lwd = 2, col = "red")
  hist(x = vparms[,3], probability = TRUE, xlab = "b", main = "")
  lines(density(vparms[,3]),lwd = 2, col = "red")
  hist(x = vparms[,4], probability = TRUE, xlab = "rho", main = "")
  lines(density(vparms[,4]),lwd = 2, col = "red")
  titre <- paste("alpha = ", alpha[i], ", beta = ", beta[i], ", b = ", b[1], ", rho = ", rho[i], sep = "")
  title(main = titre, line = -3, outer = TRUE)
}

dev.off()
