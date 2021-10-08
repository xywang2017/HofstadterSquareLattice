using LinearAlgebra
using PyPlot
using Parameters
using JLD
using Printf
fpath="/Users/xiaoyuw/Desktop/HofstadterSquareLattice/"
include(joinpath(fpath,"libs/HofstadterLL.jl"))

##
qs = collect(2:40)
p = 1
data = Dict()
for iq in eachindex(qs)
    println("q=$(qs[iq])")
    q = qs[iq]
    hof = constructLLHofstadter(q=q,p=p,nLL=20)
    data["$iq"] =  hof.spectrum[:]
end
save("LL_results.jld","data",data)

##
data=load("LL_results.jld","data")
fig = figure()
for iq in eachindex(qs)
    q = qs[iq]
    ϵ = data["$iq"]
    ϕ = ones(size(ϵ)) * p/q
    plot(ϕ,ϵ,"b.",ms=0.5,markeredgecolor="none")
end
tight_layout()
display(fig)
close(fig)

##
hof = constructLLHofstadter(q=12,p=1,nLL=5);