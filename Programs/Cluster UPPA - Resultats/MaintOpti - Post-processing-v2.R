#rm(list=ls())

library(readxl)
library(writexl)

MatParam <- read_xlsx(path = "../../Figures2/res.xlsx")
Case <- "H"
idx.case <- which(MatParam$Classe == Case)
nb.files <- length(idx.case)

Ficname <- paste("Results-cas-", idx.case[1], ".Rd", sep = "")
load(Ficname)
Tab <- output[[1]]

if (nb.files>1) {
  for (i in 2:nb.files) {
    Ficname <- paste("Results-cas-", idx.case[i], ".Rd", sep = "")
    load(Ficname)
    Tab <- cbind(Tab, output[[1]]$cout.L)
  }
}

xmin = min(Tab[, 1])
xmax = max(Tab[, 1])
ymin = min(Tab[, -1])
ymax = max(Tab[, -1])

Titre <- paste("../../Figures2/All-cases-", Case, ".pdf", sep = "")

pdf(Titre)

plot(x = Tab[, 1], y = Tab[, 2], type = "l", lwd = 2, xlim = c(xmin, xmax), ylim = c(ymin, ymax))
if (nb.files>1) {
  for (i in 2:nb.files) {
    lines(x = Tab[, 1], y = Tab[, i+1], lwd = 2)
  }
}

dev.off()
