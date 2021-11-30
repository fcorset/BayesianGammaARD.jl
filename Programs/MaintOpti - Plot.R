rm(list=ls())

load(file = "Resultat.Rd")

plot(x = output$seq.L, y = output$cout.L, type = "l", lwd = 3, xlab ="L", ylab = "Cost",
     main = "Cost for (C_I,C_P,C_C)=(1,5,10) and tau = 10")

cout.L.smooth <- loess(formula = cout.L ~ seq.L, span = 0.8, data = output)

xfit <- seq(from = min(output$seq.L), to = max(output$seq.L), by = 0.01)
yfit1 <- predict(cout.L.smooth, newdata = xfit)
lines(x = xfit, y = yfit1, col = "red", lwd = 3, xlab = "L")
idx <- which.min(yfit1)
cat("Coût optimal : ", yfit1[idx], "\n")
cat("L optimal : ", xfit[idx], "\n")

