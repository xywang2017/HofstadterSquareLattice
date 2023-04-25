using LinearAlgebra
using PyPlot
using Parameters
using JLD
using Printf
fpath=pwd()
include(joinpath(fpath,"libs/HofstadterLL.jl"))

##
qs = collect(1:20)
p = 1
data = Dict()
for iq in eachindex(qs)
    println("q=$(qs[iq])")
    q = qs[iq]
    hof = constructLLHofstadter(q=q,p=p,lk=q,nLL=max(50,10q),V0=3.0)
    data["$iq"] =  hof.spectrum[:]
end
save("LL_results.jld","data",data)

##

fig = figure(figsize=(4,3))

# data = load("Q64_results.jld","data")
# ps = 0:64
# for ip in eachindex(ps)
#     p = ps[ip]
#     ϵ = data["$ip"]
#     ϕ = ones(size(ϵ)) * p/64
#     plot(ϕ,ϵ,"b.",ms=1,markeredgecolor="none")
# end

# data = load("Q27_results.jld","data")
# ps = 0:27
# for ip in eachindex(ps)
#     p = ps[ip]
#     ϵ = data["$ip"]
#     ϕ = ones(size(ϵ)) * p/27
#     plot(ϕ,ϵ,"b.",ms=1,markeredgecolor="none")
# end

data=load("LL_results.jld","data")
for iq in eachindex(qs)
    p = 1
    q = qs[iq]
    ϵ = data["$iq"]
    ϕ = ones(size(ϵ)) * p/q
    plot(ϕ,ϵ,"r.",ms=1,markeredgecolor="none")
end

ylabel(L"ϵ/ϵ_0")
xlabel(L"ϕ/ϕ_0")
ylim([-0.1,0.3])
tight_layout()
display(fig)
savefig("LLhofstadter.png",dpi=330)
close(fig)

##
hof = constructLLHofstadter(q=2,p=1,nLL=40);