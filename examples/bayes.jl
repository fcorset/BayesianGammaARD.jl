using BayesianGammaARD
using Distributions
ni = NonInformative(:θ)

pdf(ni,2)



pdf(InverseGamma(),2)
ni isa Distribution

