library(DescTools)

epsilon <- 0

alpha <- rep(x = 1, times = 9)
beta <- c(1, 1, 1, 0.75, 0.75, 0.75, 1.2, 1.2, 1.2)
b <- rep(x = 1, times = 9)
rho <- c(0.2, 0.5, 0.8, 0.2, 0.5, 0.8, 0.2, 0.5, 0.8)
TabCas <- cbind(alpha, beta, b, rho)

ListFic <- list.files()
idx <- grep(pattern = "Rd", x = ListFic)
ListFic <- ListFic[idx]

NbFic <- length(x = ListFic)


for (i in 1:NbFic) {
  Fic <- ListFic[i]
  load(file = Fic)
  NumCas <- as.integer(strsplit(Fic,"")[[1]][11])

  vparms <- res[[2]]
  
  idx <- which((vparms[, 1] < 0) | (vparms[, 2] < 0) | (vparms[, 3] < 0) | (vparms[, 4] < 0) | (vparms[, 4] > 1))
  if (length(idx)>0) {
    vparms <- vparms[-idx, ]
  }
  
  FicOut <- paste("Plot-cas",NumCas,".pdf", sep = "")
  pdf(FicOut)
  
  # PLOT 1
  aux <- Winsorize(vparms[,1], probs = c(epsilon, 1-epsilon))
  par(mfrow = c(2,2))
  f <- density(aux)
  h <- hist(x = aux, plot = FALSE)
  ymax <- max(c(f$y, h$density))
  hist(x = aux, probability = TRUE, xlab = expression(widehat(alpha)), main = "", ylim = c(0, ymax))
  lines(f,lwd = 2, col = "red")
  abline(v = alpha, lty = 2, lwd = 2, col = "blue")

  aux <- Winsorize(vparms[,2], probs = c(epsilon, 1-epsilon))
  f <- density(aux)
  h <- hist(x = aux, plot = FALSE)
  ymax <- max(c(f$y, h$density))
  hist(x = aux, probability = TRUE, xlab = expression(widehat(beta)), main = "", ylim = c(0, ymax))
  lines(f,lwd = 2, col = "red")
  abline(v = beta[NumCas], lty = 2, lwd = 2, col = "blue")

  aux <- Winsorize(vparms[,3], probs = c(epsilon, 1-epsilon))
  f <- density(aux)
  h <- hist(x = aux, plot = FALSE)
  ymax <- max(c(f$y, h$density))
  hist(x = aux, probability = TRUE, xlab = expression(widehat(b)), main = "", ylim = c(0, ymax))
  lines(f,lwd = 2, col = "red")
  abline(v = b, lty = 2, lwd = 2, col = "blue")

  aux <- Winsorize(vparms[,4], probs = c(epsilon, 1-epsilon))
  f <- density(aux)
  h <- hist(x = aux, plot = FALSE)
  ymax <- max(c(f$y, h$density))
  hist(x = aux, probability = TRUE, xlab = expression(widehat(rho)), main = "", ylim = c(0, ymax))
  lines(f,lwd = 2, col = "red")
  abline(v = rho[NumCas], lty = 2, lwd = 2, col = "blue")
  titre <- paste("alpha = ", alpha[i], ", beta = ", beta[i], ", b = ", b[1], ", rho = ", rho[i], sep = "")
  titre <- expression(paste(alpha, "=5, ", beta, "=5, ",b, "=5, ",rho, "=1"))
  if (NumCas == 1) {
    titre <- expression(paste("Case 1: ", alpha, "= 1.0, ", beta, "= 1.0, ",b, "= 1.0, ",rho, "= 0.2"))
  }
  if (NumCas == 2) {
    titre <- expression(paste("Case 2: ", alpha, "= 1.0, ", beta, "= 1.0, ",b, "= 1.0, ",rho, "= 0.5"))
  }
  if (NumCas == 3) {
    titre <- expression(paste("Case 3: ", alpha, "= 1.0, ", beta, "= 1.0, ",b, "= 1.0, ",rho, "= 0.8"))
  }
  if (NumCas == 4) {
    titre <- expression(paste("Case 4: ", alpha, "= 1.0, ", beta, "= 0.75, ",b, "= 1.0, ",rho, "= 0.2"))
  }
  if (NumCas == 5) {
    titre <- expression(paste("Case 5: ", alpha, "= 1.0, ", beta, "= 0.75, ",b, "= 1.0, ",rho, "= 0.5"))
  }
  if (NumCas == 6) {
    titre <- expression(paste("Case 6: ", alpha, "= 1.0, ", beta, "= 0.75, ",b, "= 1.0, ",rho, "= 0.8"))
  }
  if (NumCas == 7) {
    titre <- expression(paste("Case 7: ", alpha, "= 1.0, ", beta, "= 1.2, ",b, "= 1.0, ",rho, "= 0.2"))
  }
  if (NumCas == 8) {
    titre <- expression(paste("Case 8: ", alpha, "= 1.0, ", beta, "= 1.2, ",b, "= 1.0, ",rho, "= 0.5"))
  }
  if (NumCas == 9) {
    titre <- expression(paste("Case 9: ", alpha, "= 1.0, ", beta, "= 1.2, ",b, "= 1.0, ",rho, "= 0.8"))
  }
  title(main = titre, line = -3, outer = TRUE)
  # PLOT 2
  aux <- vparms
  vparms[, 1:3] <- log(vparms[, 1:3])
  vparms[, 4] <- qnorm(vparms[, 4])
  vparms <- apply(X = aux, MARGIN = 2, scale)
  par(mfrow = c(2,2))
  hist(x = vparms[,1], probability = TRUE, xlab = expression(widehat(alpha)), main = "")
  lines(density(vparms[,1]),lwd = 2, col = "red")
  abline(v = log(alpha), lty = 2, lwd = 2, col = "blue")
  hist(x = vparms[,2], probability = TRUE, xlab = expression(widehat(beta)), main = "")
  lines(density(vparms[,2]),lwd = 2, col = "red")
  abline(v = log(beta[i]), lty = 2, lwd = 2, col = "blue")
  hist(x = vparms[,3], probability = TRUE, xlab = expression(widehat(b)), main = "")
  lines(density(vparms[,3]),lwd = 2, col = "red")
  abline(v = log(b), lty = 2, lwd = 2, col = "blue")
  hist(x = vparms[,4], probability = TRUE, xlab = expression(widehat(rho)), main = "")
  lines(density(vparms[,4]),lwd = 2, col = "red")
  abline(v = qnorm(rho[i]), lty = 2, lwd = 2, col = "blue")
  titre <- paste("alpha = ", alpha[i], ", beta = ", beta[i], ", b = ", b[1], ", rho = ", rho[i], sep = "")
  titre <- expression(paste(alpha, "=5, ", beta, "=5, ",b, "=5, ",rho, "=1"))
  if (NumCas == 1) {
    titre <- expression(paste("Case 1 (transformed data): ", alpha, "= 1.0, ", beta, "= 1.0, ",b, "= 1.0, ",rho, "= 0.2"))
  }
  if (NumCas == 2) {
    titre <- expression(paste("Case 2 (transformed data): ", alpha, "= 1.0, ", beta, "= 1.0, ",b, "= 1.0, ",rho, "= 0.5"))
  }
  if (NumCas == 3) {
    titre <- expression(paste("Case 3 (transformed data): ", alpha, "= 1.0, ", beta, "= 1.0, ",b, "= 1.0, ",rho, "= 0.8"))
  }
  if (NumCas == 4) {
    titre <- expression(paste("Case 4 (transformed data): ", alpha, "= 1.0, ", beta, "= 0.75, ",b, "= 1.0, ",rho, "= 0.2"))
  }
  if (NumCas == 5) {
    titre <- expression(paste("Case 5 (transformed data): ", alpha, "= 1.0, ", beta, "= 0.75, ",b, "= 1.0, ",rho, "= 0.5"))
  }
  if (NumCas == 6) {
    titre <- expression(paste("Case 6 (transformed data): ", alpha, "= 1.0, ", beta, "= 0.75, ",b, "= 1.0, ",rho, "= 0.8"))
  }
  if (NumCas == 7) {
    titre <- expression(paste("Case 7 (transformed data): ", alpha, "= 1.0, ", beta, "= 1.2, ",b, "= 1.0, ",rho, "= 0.2"))
  }
  if (NumCas == 8) {
    titre <- expression(paste("Case 8 (transformed data): ", alpha, "= 1.0, ", beta, "= 1.2, ",b, "= 1.0, ",rho, "= 0.5"))
  }
  if (NumCas == 9) {
    titre <- expression(paste("Case 9 (transformed data): ", alpha, "= 1.0, ", beta, "= 1.2, ",b, "= 1.0, ",rho, "= 0.8"))
  }
  title(main = titre, line = -3, outer = TRUE)
  # PLOT 3
  vparms <- apply(X = vparms, MARGIN = 2, scale)
  par(mfrow = c(2,2))
  hist(x = vparms[,1], probability = TRUE, xlab = expression(widehat(alpha)), main = "")
  lines(density(vparms[,1]),lwd = 2, col = "red")
  hist(x = vparms[,2], probability = TRUE, xlab = expression(widehat(beta)), main = "")
  lines(density(vparms[,2]),lwd = 2, col = "red")
  hist(x = vparms[,3], probability = TRUE, xlab = expression(widehat(b)), main = "")
  lines(density(vparms[,3]),lwd = 2, col = "red")
  hist(x = vparms[,4], probability = TRUE, xlab = expression(widehat(rho)), main = "")
  lines(density(vparms[,4]),lwd = 2, col = "red")
  titre <- paste("alpha = ", alpha[i], ", beta = ", beta[i], ", b = ", b[1], ", rho = ", rho[i], sep = "")
  titre <- expression(paste(alpha, "=5, ", beta, "=5, ",b, "=5, ",rho, "=1"))
  if (NumCas == 1) {
    titre <- expression(paste("Case 1 (transformed and normalized data): ", alpha, "= 1.0, ", beta, "= 1.0, ",b, "= 1.0, ",rho, "= 0.2"))
  }
  if (NumCas == 2) {
    titre <- expression(paste("Case 2 (transformed and normalized data): ", alpha, "= 1.0, ", beta, "= 1.0, ",b, "= 1.0, ",rho, "= 0.5"))
  }
  if (NumCas == 3) {
    titre <- expression(paste("Case 3 (transformed and normalized data): ", alpha, "= 1.0, ", beta, "= 1.0, ",b, "= 1.0, ",rho, "= 0.8"))
  }
  if (NumCas == 4) {
    titre <- expression(paste("Case 4 (transformed and normalized data): ", alpha, "= 1.0, ", beta, "= 0.75, ",b, "= 1.0, ",rho, "= 0.2"))
  }
  if (NumCas == 5) {
    titre <- expression(paste("Case 5 (transformed and normalized data): ", alpha, "= 1.0, ", beta, "= 0.75, ",b, "= 1.0, ",rho, "= 0.5"))
  }
  if (NumCas == 6) {
    titre <- expression(paste("Case 6 (transformed and normalized data): ", alpha, "= 1.0, ", beta, "= 0.75, ",b, "= 1.0, ",rho, "= 0.8"))
  }
  if (NumCas == 7) {
    titre <- expression(paste("Case 7 (transformed and normalized data): ", alpha, "= 1.0, ", beta, "= 1.2, ",b, "= 1.0, ",rho, "= 0.2"))
  }
  if (NumCas == 8) {
    titre <- expression(paste("Case 8 (transformed and normalized data): ", alpha, "= 1.0, ", beta, "= 1.2, ",b, "= 1.0, ",rho, "= 0.5"))
  }
  if (NumCas == 9) {
    titre <- expression(paste("Case 9 (transformed and normalized data): ", alpha, "= 1.0, ", beta, "= 1.2, ",b, "= 1.0, ",rho, "= 0.8"))
  }
  title(main = titre, line = -3, outer = TRUE)

  dev.off()
}

