using BayesianGammaARD
using Distributions
ni = NonInformative(:θ)

pdf(ni,3)
c=1.2
d=1.5
dd = Informative(:θ,c,d)
params(dd)
dd isa Distribution

pdf(dd,2)

ni isa Distribution
