using LinearAlgebra
using PyPlot
using Parameters
using JLD
using Printf
fpath="/Users/xiaoyu/Code/HofstadterSquareLattice/"
include(joinpath(fpath,"libs/Hofstadter.jl"))

##
q = 25
ps = collect(0:q)
data = Dict()
for ip in eachindex(ps)
    @printf("p/q=%d/%d\n",ps[ip],q);
    h0,hof = computeHofstadter(p=ps[ip],q=q);
    data["$ip"] = hof.spectrum[:];
end
save("Q$(q)_results.jld","data",data)

##
fig = figure(figsize=(4,3))
colors = ["b","b","b","b","b"]
qs = [25]
for iq in eachindex(qs)
    q = qs[iq]
    data = load("Q$(q)_results.jld","data")
    ps = collect(0:q)
    for ip in eachindex(ps)
        ϵ = data["$ip"]
        plot(ones(size(ϵ))*ps[ip]/q,ϵ,".",c=colors[iq],ms=1.5,markerfacecolor="b",markeredgecolor="none")
    end
end
ylabel(L"ϵ")
xlabel(L"ϕ/ϕ_0")
tight_layout()
savefig("energy_flux.pdf")
display(fig)
close(fig)

##
# real space 
A = initHamiltonian(lk=25,V0=1.0)
function Ψr(z::ComplexF64,h0::Hamiltonian,R::Vector{Int})
    # R = [m,n]a
    zR = (R[1] + 1im * R[2])*h0.a
    Ψk = reshape(h0.Uk,h0.lg^2,:)
    expigr = reshape(exp.(1im * real(h0.gvec .* z') ),:,1)
    expikr = reshape(exp.(1im * real(h0.kvec .* z') ),1,:)
    expikR = reshape(exp.(-1im * real(h0.kvec .* zR') ),1,:)
    return sum(expigr .* Ψk .* expikr .* expikR)/sqrt(h0.l1*h0.l2)
end

area = -2:0.1:2
zgrid = reshape(area,:,1) .+ 1im * reshape(area,1,:)
z = zgrid[:]
ΨRr = zeros(ComplexF64,length(z))
for iz in eachindex(z) 
    ΨRr[iz] = Ψr(z[iz],A,[0,0])
end

##
fig = figure(figsize=(4,3))
weight = reshape(abs.(ΨRr).^2,length(area),length(area))
pl=contourf(real(zgrid),imag(zgrid),weight,cmap="Blues_r")
colorbar(pl)
axis("equal")
tight_layout()
display(fig)
close(fig)