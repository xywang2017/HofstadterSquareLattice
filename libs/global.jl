using Arpack 

mutable struct Hamiltonian 
    a::Float64
    g::Float64
    m::Float64
    V0::Float64   # 2V0 ( cos(g1 x) + cos(g2 y) )
    l1::Int
    l2::Int 
    gvec::Matrix{ComplexF64}
    kvec::Matrix{ComplexF64}
    k1::Vector{Float64}
    k2::Vector{Float64}
    lg::Int
    nbands::Int
    Uk::Array{Float64,4}  # U_g(n,k1,k2)
    Hk::Array{Float64,3}

    Hamiltonian() = new()
end

function initHamiltonian(;lg::Int=11,lk::Int=40,a::Float64=1.0,m::Float64=1.0,V0::Float64=1.0,nbands::Int=1)
    A = Hamiltonian()
    A.a = a 
    A.g = 2π/A.a
    A.m = m 
    A.V0 = V0 
    A.l1 = lk 
    A.l2 = lk 
    A.lg = lg 
    A.nbands = nbands

    A.k1 = collect(0:(A.l1-1)) ./ A.l1 
    A.k2 = collect(0:(A.l2-1)) ./ A.l2
    A.kvec = (reshape(A.k1,:,1) .+ 1im*reshape(A.k2,1,:)) * A.g
    g0 = (-(lg-1)÷2):((lg-1)÷2)
    A.gvec = (reshape(g0,:,1) .+ 1im* reshape(g0,1,:)) * A.g
    A.Uk = zeros(Float64,lg^2,nbands,A.l1,A.l2)
    A.Hk = zeros(Float64,nbands,A.l1,A.l2)

    for ik in eachindex(A.kvec[:])
        ComputeEigenSpectrum(A,ik)
    end

    ## smooth gauge 
    Uk = reshape(A.Uk,lg^2,A.l1,A.l2)
    for ik2 in 1:size(A.kvec,2)
        for ik1 in 1:(size(A.kvec,1)-1)
            tmp = Uk[:,ik1,ik2]' * Uk[:,ik1+1,ik2]
            if tmp < 0
                Uk[:,ik1+1,ik2] = - Uk[:,ik1+1,ik2]
            end    
        end
    end
    for ik2 in 1:(size(A.kvec,2)-1)
        tmp = Uk[:,1,ik2]' * Uk[:,1,ik2+1]
        if tmp < 0
            Uk[:,:,ik2+1] = - Uk[:,:,ik2+1]
        end
    end
    
    return A
end

function ComputeEigenSpectrum(A::Hamiltonian,ik::Int)
    H = zeros(Float64,A.lg^2,A.lg^2)

    k = view(A.kvec,:)[ik]

    gvec = reshape(A.gvec,:)
    for ig in eachindex(gvec)
        H[ig,ig] = abs(k+gvec[ig])^2/(2*A.m)

        ig1 = mod(ig-1,A.lg) + 1
        ig2 = (ig-1)÷A.lg + 1 

        ig1nn = mod(ig1,A.lg) + 1
        ignn = ig1nn + (ig2-1)*A.lg
        H[ig,ignn] = -A.V0 

        ig1nn = mod(ig1-2,A.lg) + 1
        ignn = ig1nn + (ig2-1)*A.lg
        H[ig,ignn] = -A.V0 

        ig2nn = mod(ig2,A.lg) + 1
        ignn = ig1 + (ig2nn-1)*A.lg
        H[ig,ignn] = -A.V0 

        ig2nn = mod(ig2-2,A.lg) + 1
        ignn = ig1 + (ig2nn-1)*A.lg
        H[ig,ignn] = -A.V0 
    end 

    if (norm(H - H')>1e-6)
        println("error with Hermiticity")
    end

    i1 = mod(ik-1,A.l1) + 1
    i2 = (ik-1)÷A.l1 + 1
    A.Hk[:,i1,i2], A.Uk[:,:,i1,i2] = eigs(Hermitian(H),nev=A.nbands,which=:SM)
    
    return nothing
end