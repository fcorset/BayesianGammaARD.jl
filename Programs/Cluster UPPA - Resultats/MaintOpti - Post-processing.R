#rm(list=ls())

library(readxl)
library(writexl)

# 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
# 17 18 19 20 21 22 23 24 25 26
# 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48
# 50 51

NumCas <- 4
fic.name <- paste("Results-cas-", NumCas, ".Rd", sep = "")

load(file = fic.name)

fig.ficname <- paste("../../Figures2/Results-cas-", NumCas, ".pdf", sep = "")
pdf(fig.ficname)
plot(x = output[[1]]$seq.L, y = output[[1]]$cout.L, type = "l", lwd = 3, xlab ="L", ylab = "Cost", main = "")

cout.L.smooth <- loess(formula = cout.L ~ seq.L, span = 0.5, data = output[[1]])

xfit <- seq(from = min(output[[1]]$seq.L), to = max(output[[1]]$seq.L), by = 0.01)
yfit1 <- predict(cout.L.smooth, newdata = xfit)
lines(x = xfit, y = yfit1, col = "red", lwd = 3, xlab = "L")
idx <- which.min(yfit1)
cat("Coût optimal : ", yfit1[idx], "\n")
cat("L optimal : ", xfit[idx], "\n")

dev.off()


MatParam <-  read_excel("../../Figures2/res.xlsx")
MatParam$L_opt[NumCas] <- xfit[idx]
MatParam$Cout_opt[NumCas] <- yfit1[idx]

write_xlsx(x = MatParam, path = "../../Figures2/res.xlsx")


