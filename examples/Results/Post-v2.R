library(readr)
library(ggplot2)
library(gridExtra)
library(cowplot)

nb.dec <- 2

# Cas Concave : β = 0.7
# Vraies valeurs des paramètres
# αtrue = 2.0
# βtrue = 0.7
# θtrue = 2.0
# ρtrue = 0.7 et ρtrue = 0.2

# Case <- "Concave"
# Case <- "Convex"
Case <- "Homogeneous"
File.Path <- paste("./", Case, "/", sep = "")

setwd(File.Path)
alpha.true = 2.0
beta.true = 0.7
theta.true = 2.0
# ρtrue = 0.7 et ρtrue = 0.2
LF <- list.files(path = "./", pattern = "bayes")
nbf <- length(LF)

Df.MLE.rho0.2 <- read_csv(file = "mle_results_rho0.2.csv")
Df.MLE.rho0.7 <- read_csv(file = "mle_results_rho0.7.csv")

for (i in 1:nbf) {
  FicName <- LF[i]
  Df <- read_csv(FicName)
  
  is.bad.prior <- (length(grep(pattern = "bad", x = FicName))>0)
  
  is.noninformative.prior <- (length(grep(pattern = "non_informative", x = FicName))>0)
  
  tmp <- strsplit(x = FicName, "_")
  rho.true <- tmp[[1]][6]
  if (is.na(rho.true)) {
    next
  }
  rho.true <- strsplit(x = rho.true, split = "")
  rho.true <- rho.true[[1]][4:6]
  rho.true <- as.numeric(paste0(rho.true, collapse = ""))
  if (rho.true == 0.2) {
    Df.MLE <- Df.MLE.rho0.2 
  }
  if (rho.true == 0.7) {
    Df.MLE <- Df.MLE.rho0.7
  }
  
  n <- tmp[[1]][5]
  n <- strsplit(x = n, split = "")
  n <- n[[1]][-1]
  n <- as.numeric(paste0(n, collapse = ""))
  idx <- which(Df.MLE$n == n)
  Bayes.Mean <- colMeans(Df)
  
  Prior.dist <- vector(mode = "list", length = 4)
  
  # alpha
  w <- 1
  priormeanEw <- theta.true * alpha.true * w^beta.true
  priorvarEw <- 0.2
  f <- priorvarEw / priormeanEw
  e <- priormeanEw / f
  x.alpha.min <- 0
  x.alpha.max <- round(max(Df$α)*1.2, digits = nb.dec)
  x.alpha <- seq(from = x.alpha.min, by = 10^(-nb.dec), to = x.alpha.max)
  y.alpha <- dgamma(x = x.alpha, shape = e, scale = f)
  Prior.dist[[1]] <- data.frame(x.alpha, y.alpha)
  
  # beta
  if (Case == "Concave") {
    beta.priormean <- beta.true
    beta.priorvar <- 0.1
    c <- beta.priorvar /(beta.priormean-beta.priormean^2 - beta.priorvar/beta.priormean)
    d <- c * (1/beta.priormean - 1)
    x.beta <- seq(from = 0, by = 10^(-nb.dec), to = 1)
    y.beta <- dbeta(x = x.beta, shape1 = c, shape2 = d)
  }
  if (Case == "Convex") {
    beta.priormean <- 1.5
    beta.priorvar <- 0.5
    d <- beta.priorvar / (beta.priormean - 1)
    c <- (beta.priormean - 1) / d
    x.beta <- seq(from = 1, by = 10^(-nb.dec), to = max(Df$β)*1.2)
    y.beta <- dgamma(x = x.beta - 1, shape = c, scale = d)
  }
  if (Case == "Homogeneous") {
    c <- 2
    x.beta.min <- round(min(Df$β)*0.8, digits = nb.dec)
    x.beta.min <- 0
    x.beta.max <- round(max(Df$β)*1.2, digits = nb.dec)
    x.beta <- seq(from = x.beta.min, by = 10^(-nb.dec), to = x.beta.max)
    y.beta <- dgamma(x = x.beta, shape = c, rate = c)
  }
  Prior.dist[[2]] <- data.frame(x.beta, y.beta)
  
  # theta
  priormean.theta <- theta.true
  priorvar.theta <- 0.2
  a <- 2 + priormean.theta^2 / priorvar.theta
  b <- (a-1) * priormean.theta
  x.theta.min <- 0
  x.theta.max <- round(max(Df$θ)*1.2, digits = nb.dec)
  x.theta <- seq(from = x.theta.min, by = 10^(-nb.dec), to = x.theta.max)
  y.theta <- b^a/gamma(a)/x.theta^(a+1)*exp(-b/x.theta)
  Prior.dist[[3]] <- data.frame(x.theta, y.theta)
  
  # rho
  
  
  gr1 <- ggplot(data = Df, aes(x = α)) + 
    geom_density() +
    xlab(expression(alpha)) +
    ylab("") +
    geom_vline(aes(xintercept = alpha.true), 
               color = "red", linetype="dashed") +
    geom_vline(aes(xintercept = Df.MLE$α[idx]), 
               color = "blue", linetype="dashed") +
    geom_vline(aes(xintercept = Bayes.Mean[1]), 
             color = "black", linetype="dashed") +
    theme_classic()
  if ((!is.noninformative.prior) & (!is.bad.prior)) {
    gr1 <- gr1
    gr1 <- gr1 +
      geom_line(data = Prior.dist[[1]], mapping = aes(x = x.alpha, y = y.alpha), col = grey(0.8))
  }
  
  gr2 <- ggplot(data = Df, aes(x = β)) + 
    geom_density() + 
    xlab(expression(beta)) +
    ylab("") +
    geom_vline(aes(xintercept = beta.true), 
               color = "red", linetype="dashed") +
    geom_vline(aes(xintercept = Df.MLE$β[idx]), 
               color = "blue", linetype="dashed") +
    geom_vline(aes(xintercept = Bayes.Mean[2]), 
               color = "black", linetype="dashed") +
    theme_classic()
  if ((!is.noninformative.prior) & (!is.bad.prior)) {
    gr2 <- gr2 +
      geom_line(data = Prior.dist[[2]], mapping = aes(x = x.beta, y = y.beta), col = grey(0.8))
  }
  
  gr3 <- ggplot(data = Df, aes(x = θ)) + 
    geom_density() +
    xlab(expression(theta)) +
    ylab("") +
    geom_vline(aes(xintercept = theta.true), 
               color = "red", linetype="dashed") +
    geom_vline(aes(xintercept = Df.MLE$θ[idx]), 
               color = "blue", linetype="dashed") +
    geom_vline(aes(xintercept = Bayes.Mean[3]), 
               color = "black", linetype="dashed") +
    theme_classic()
  if ((!is.noninformative.prior) & (!is.bad.prior)) {
    gr3 <- gr3
    gr3 <- gr3 +
      geom_line(data = Prior.dist[[3]], mapping = aes(x = x.theta, y = y.theta), col = grey(0.8))
  }
  
  gr4 <- ggplot(data = Df, aes(x = ρ)) + 
    geom_density() +
    xlab(expression(rho)) +
    ylab("") +
    geom_vline(aes(xintercept = rho.true), 
               color = "red", linetype="dashed") +
    geom_vline(aes(xintercept = Df.MLE$ρ[idx]), 
               color = "blue", linetype="dashed") +
    geom_vline(aes(xintercept = Bayes.Mean[4]), 
               color = "black", linetype="dashed") +
    theme_classic()
  if ((!is.noninformative.prior) & (!is.bad.prior)) {
    gr4 <- gr4
    # gr4 <- gr4 +
    #   geom_line(data = Prior.dist[[4]], mapping = aes(x = x.rho, y = y.rho), col = grey(0.8))
  }
  
  Out <- paste("./fig/",FicName, ".pdf", sep = "")
  g <- plot_grid(gr1, gr2, gr3, gr4, ncol = 2, nrow = 2)
  pdf(Out)
  plot(g)
  dev.off()
}
setwd("../")


