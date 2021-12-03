rm(list=ls())

library(readxl)
library(writexl)

NumCas <- 1
fic.name <- paste("Results-cas-", NumCas, ".Rd", sep = "")

load(file = fic.name)

plot(x = output$seq.L, y = output$cout.L, type = "l", lwd = 3, xlab ="L", ylab = "Cost", main = "")

cout.L.smooth <- loess(formula = cout.L ~ seq.L, span = 0.8, data = output)

xfit <- seq(from = min(output$seq.L), to = max(output$seq.L), by = 0.01)
yfit1 <- predict(cout.L.smooth, newdata = xfit)
lines(x = xfit, y = yfit1, col = "red", lwd = 3, xlab = "L")
idx <- which.min(yfit1)
cat("Coût optimal : ", yfit1[idx], "\n")
cat("L optimal : ", xfit[idx], "\n")

MatParam <-  read_excel("../Figures/res.xlsx")
MatParam$L_opt[NumCas] <- xfit[idx]
MatParam$Cout_opt[NumCas] <- yfit1[idx]

write_xlsx(x = MatParam, path = "../Figures/res.xlsx")


