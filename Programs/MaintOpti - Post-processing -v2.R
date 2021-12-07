#rm(list=ls())

library(readxl)
library(writexl)


LF <- list.files()
idx <- grep(pattern = "Results-cas", x = LF)
LF <- LF[idx]
nb.files <- length(LF)

load(file = LF[1])
Costs <- output[[1]]

for (i in 2:nb.files) {
  load(file = LF[i])
  Costs <- cbind.data.frame(Costs, output[[1]]$cout.L)
}

ymin = min(Costs[, -1])
ymax = max(Costs[, -1])

plot(x = Costs[, 1], y = Costs[, 2], type = "l", lwd = 2, ylim = c(ymin, ymax))
for (i in 2:nb.files) {
  lines(x = Costs[, 1], y = Costs[, i+1], type = "l", lwd = 2)
}


