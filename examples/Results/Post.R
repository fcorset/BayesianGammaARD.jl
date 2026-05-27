library(readr)
library(ggplot2)
library(gridExtra)
library(cowplot)

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
  
  if (Case == "Concave") {
    beta.priormean <- beta.true
    beta.priorvar <- 0.1
    c <- beta.priorvar /(beta.priormean-beta.priormean^2 - beta.priorvar/beta.priormean)
    d <- c * (1/beta.priormean - 1)
    nb.dec <- 2
    x.beta <- seq(from = 0, by = 10^(-nb.dec), to = 1)
    y.beta <- dbeta(x = x.beta, shape1 = c, shape2 = d)
  }
  if (Case == "Convex") {
    beta.priormean <- 1.5
    beta.priorvar <- 0.5
    d <- beta.priorvar / (beta.priormean - 1)
    c <- (beta.priormean - 1) / d
  }
  if (Case == "Homogeneous") {
    
  }
  Prior.dist <- data.frame(x.beta, y.beta)
  
  
  gr1 <- ggplot(data = Df, aes(x = α)) + 
    geom_density() +
    xlab(expression(alpha)) +
    ylab("") +
    geom_vline(aes(xintercept = alpha.true), 
               color = "red", linetype="dashed") +
    geom_vline(aes(xintercept = Df.MLE$α[idx]), 
               color = "blue", linetype="dashed") +
    geom_vline(aes(xintercept = Bayes.Mean[1]), 
             color = "black", linetype="dashed")
  gr2 <- ggplot(data = Df, aes(x = β)) + 
    geom_density() + 
    xlab(expression(beta)) +
    ylab("") +
    geom_vline(aes(xintercept = beta.true), 
               color = "red", linetype="dashed") +
    geom_vline(aes(xintercept = Df.MLE$β[idx]), 
               color = "blue", linetype="dashed") +
    geom_vline(aes(xintercept = Bayes.Mean[2]), 
               color = "black", linetype="dashed") 
    # geom_line(data = Prior.dist, 
    #           mapping = aes(x = x.beta, y = y.beta), col = grey(0.8)) +
    theme_classic()
  gr3 <- ggplot(data = Df, aes(x = θ)) + 
    geom_density() +
    xlab(expression(theta)) +
    ylab("") +
    geom_vline(aes(xintercept = theta.true), 
               color = "red", linetype="dashed") +
    geom_vline(aes(xintercept = Df.MLE$θ[idx]), 
               color = "blue", linetype="dashed") +
    geom_vline(aes(xintercept = Bayes.Mean[3]), 
               color = "black", linetype="dashed")
  gr4 <- ggplot(data = Df, aes(x = ρ)) + 
    geom_density() +
    xlab(expression(rho)) +
    ylab("") +
    geom_vline(aes(xintercept = rho.true), 
               color = "red", linetype="dashed") +
    geom_vline(aes(xintercept = Df.MLE$ρ[idx]), 
               color = "blue", linetype="dashed") +
    geom_vline(aes(xintercept = Bayes.Mean[4]), 
               color = "black", linetype="dashed")
  
  Out <- paste("./fig/",FicName, ".pdf", sep = "")
  g <- plot_grid(gr1, gr2, gr3, gr4, ncol = 2, nrow = 2)
  pdf(Out)
  plot(g)
  dev.off()
}
setwd("../")



# Cas Convex : ρ = 0.7
# Charger les données simulées en récupérant les valeurs des paramètres
# Vraies valeurs des paramètres
# αtrue = 0.8
# βtrue = 1.5
# θtrue = 1.0
# ρtrue = 0.7 et ρtrue = 0.2

setwd("./Convex/")
alpha.true = 0.8
beta.true = 1.5
theta.true = 1.0
# ρtrue = 0.7 et ρtrue = 0.2
LF <- list.files(path = "./", pattern = "bayes")
nbf <- length(LF)

Df.MLE.rho0.2 <- read_csv(file = "mle_results_rho0.2.csv")
Df.MLE.rho0.7 <- read_csv(file = "mle_results_rho0.7.csv")

for (i in 1:nbf) {
  FicName <- LF[i]
  Df <- read_csv(FicName)
  
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
  
  gr1 <- ggplot(data = Df, aes(x = α)) + 
    geom_density() +
    ylab("") +
    geom_vline(aes(xintercept = alpha.true), 
               color = "red", linetype="dashed") +
    geom_vline(aes(xintercept = Df.MLE$α[idx]), 
               color = "blue", linetype="dashed")
  gr2 <- ggplot(data = Df, aes(x = β)) + 
    geom_density() +
    ylab("") +
    geom_vline(aes(xintercept = beta.true), 
               color = "red", linetype="dashed") +
    geom_vline(aes(xintercept = Df.MLE$β[idx]), 
               color = "blue", linetype="dashed")
  gr3 <- ggplot(data = Df, aes(x = θ)) + 
    geom_density() +
    ylab("") +
    geom_vline(aes(xintercept = theta.true), 
               color = "red", linetype="dashed") +
    geom_vline(aes(xintercept = Df.MLE$θ[idx]), 
               color = "blue", linetype="dashed")
  gr4 <- ggplot(data = Df, aes(x = ρ)) + 
    geom_density() +
    ylab("") +
    geom_vline(aes(xintercept = rho.true), 
               color = "red", linetype="dashed") +
    geom_vline(aes(xintercept = Df.MLE$ρ[idx]), 
               color = "blue", linetype="dashed")
  
  Out <- paste("./fig/",FicName, ".pdf", sep = "")
  g <- plot_grid(gr1, gr2, gr3, gr4, ncol = 2, nrow = 2)
  pdf(Out)
  plot(g)
  dev.off()
}
setwd("../")


# Cas Homogeneous : β=1.0
# Vraies valeurs des paramètres
# αtrue = 0.8
# βtrue = 1.0
# θtrue = 1.0
# ρtrue = 0.7 et ρtrue = 0.2

setwd("./Homogeneous/")
alpha.true = 0.8
beta.true = 1.0
theta.true = 1.0
# ρtrue = 0.7 et ρtrue = 0.2
LF <- list.files(path = "./", pattern = "bayes")
nbf <- length(LF)

for (i in 1:nbf) {
  FicName <- LF[i]
  Df <- read_csv(FicName)
  
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
  
  gr1 <- ggplot(data = Df, aes(x = α)) + 
    geom_density() +
    ylab("") +
    geom_vline(aes(xintercept = alpha.true), 
               color = "red", linetype="dashed") +
    geom_vline(aes(xintercept = Df.MLE$α[idx]), 
               color = "blue", linetype="dashed")
  gr2 <- ggplot(data = Df, aes(x = β)) + 
    geom_density() +
    ylab("") +
    geom_vline(aes(xintercept = beta.true), 
               color = "red", linetype="dashed") +
    geom_vline(aes(xintercept = Df.MLE$β[idx]), 
               color = "blue", linetype="dashed")
  gr3 <- ggplot(data = Df, aes(x = θ)) + 
    geom_density() +
    ylab("") +
    geom_vline(aes(xintercept = theta.true), 
               color = "red", linetype="dashed") +
    geom_vline(aes(xintercept = Df.MLE$θ[idx]), 
               color = "blue", linetype="dashed")
  gr4 <- ggplot(data = Df, aes(x = ρ)) + 
    geom_density() +
    ylab("") +
    geom_vline(aes(xintercept = rho.true), 
               color = "red", linetype="dashed") +
    geom_vline(aes(xintercept = Df.MLE$ρ[idx]), 
               color = "blue", linetype="dashed")
  
  Out <- paste("./fig/",FicName, ".pdf", sep = "")
  g <- plot_grid(gr1, gr2, gr3, gr4, ncol = 2, nrow = 2)
  pdf(Out)
  plot(g)
  dev.off()
}
setwd("../")

