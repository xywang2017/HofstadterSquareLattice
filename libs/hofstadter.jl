include("global.jl");

mutable struct Hofstadter
    p::Int 
    q::Int 
    nvec::Vector{Int}
    ndim::Int
    δq2::Int

    l1::Int
    l2::Int 
    kvec::Matrix{ComplexF64}
    k1::Vector{Float64}
    k2::Vector{Float64}
    nbands::Int

    O0::Array{Float64,4}
    O1::Array{Float64,4}
    O2::Array{Float64,4}
    O3::Array{Float64,4}
    H::Array{ComplexF64,3}
    M::Array{ComplexF64,3}

    spectrum::Array{Float64,2}
    
    Hofstadter() = new()
end

function computeHofstadter(;p::Int=7,q::Int=64,ndim::Int=9)
    lk = q # along k2 direction, different values of l1 decouple
    A = initHamiltonian(lk=q,V0=3.0)
    
    B = Hofstadter()
    B.p = p 
    B.q = q
    B.ndim = ndim 
    B.nvec = collect(-(ndim-1)÷2:(ndim-1)÷2)
    B.l1 = A.l1 
    B.l2 = A.l2 
    B.k1 = A.k1 
    B.k2 = A.k2 
    B.kvec = A.kvec 
    B.nbands = A.nbands 
    B.δq2 = B.p
    
    B.O0 = zeros(Float64,B.l1,B.l1,B.l2,ndim)
    B.O1 = zeros(Float64,B.l1,B.l1,B.l2,ndim)
    B.O2 = zeros(Float64,B.l1,B.l1,B.l2,ndim)
    B.O3 = zeros(Float64,B.l1,B.l1,B.l2,ndim)

    gq = -(A.lg-1)÷2 :(A.lg-1)÷2
    k1rc = reshape(B.k1,:,1) .- reshape(B.k1,1,:)

    # Fourier transform of x/L1x, multiply by 1im at the end
    @inline function V1(q1::Float64,N1::Int)
        return abs(q1)<1e-5 ? 0.0 : cos(π*q1*N1)/(2π*q1)
    end

    # Fourier transform of (x/L1x)^2
    @inline function V2(q1::Float64,N1::Int)
        return abs(q1)<1e-5 ? N1^2/12 : cos(π*q1*N1)/(2*π^2*q1^2)
    end

    for n in eachindex(B.nvec)
        qϕ2 = B.nvec[n] * B.δq2
        tmp = eachindex(B.k2) .- qϕ2   # split into [k1-q1] + δg
        k2c = mod.(tmp .-1,B.l2) .+ 1
        δg2 = (tmp .- k2c) .÷ B.l2
        
        for i2r in eachindex(B.k2)
            i2c = k2c[i2r]
            for i1r in eachindex(B.k1) 
                i1c = i1r
                ur = view(A.Uk,:,1,i1r,i2r) 
                uc = view(A.Uk,:,1,i1c,i2c)
                uc = reshape(uc,A.lg,A.lg)
                uc = reshape(circshift(uc,(0,-δg2[i2r])),A.lg^2)
                # uc = reshape(circshift(uc,(0,0)),A.lg^2)
                λrc = ur' * uc
                B.O0[i1r,i1c,i2r,n] = λrc 
                B.O1[i1r,i1c,i2r,n] = A.Hk[1,i1r,i2r]*λrc
            end
        end

    #     for iq in eachindex(gq)
    #         q1rc = k1rc .- gq[iq] .- B.nvec[n]*B.p/(B.q)
    #         V1q = V1.(q1rc,B.l1)
    #         V2q = V2.(q1rc,B.l1)
            
            
    #         ur = view(B.Uk,:,1,:,i2r)  # blk.lg^2,hof.l1
    #         uc = view(B.Uk,:,1,:,i2c)
    #         uc = reshape(uc,A.lg,A.lg,B.l1)
    #         uc = reshape(circshift(uc,(-gq[iq],-δg2[i2r],0)),A.lg^2,B.l1)

    #         λrc = ur' * uc
    #         B.O2[:,:,i2r,n] .+= λrc.*V2q
    #         B.O3[:,:,i2r,n] .+= λrc.*V1q 
    #    end 
    end

    Hn =  zeros(ComplexF64,B.l2,B.l2,B.l1,ndim)
    B.H = zeros(ComplexF64,B.l2,B.l2,B.l1)
    B.M = zeros(ComplexF64,B.l2,B.l2,B.l1)
    for i1 in eachindex(B.k1)
        for n in eachindex(B.nvec)
            θc = reshape(exp.(-1im * 2π * B.k1 * B.nvec[n]),1,B.l1)
            θϕ = exp(1im * 2π * B.nvec[n]* B.k1[i1])
            qϕ2 = B.nvec[n] * B.δq2
            tmp = eachindex(B.k2) .- qϕ2   # split into [k1-q1] + δg
            k2c = mod.(tmp .-1,B.l2) .+ 1
            for i2r in eachindex(B.k2)
                i2c = k2c[i2r]
                B.H[i2r,i2c,i1] += sum(view(B.O1,:,:,i2r,n).*θc) * θϕ /B.l1
                B.M[i2r,i2c,i1] += sum(view(B.O0,:,:,i2r,n).*θc) * θϕ /B.l1
                Hn[i2r,i2c,i1,n] += sum(view(B.O1,:,:,i2r,n).*θc) * θϕ /B.l1
            end
        end
        B.H[:,:,i1] = (B.H[:,:,i1]+ B.H[:,:,i1]')/2
    end


    ## eigenspectrum
    B.spectrum = zeros(Float64,B.l2,B.l1)
    for i1 in eachindex(B.k1)
        M = B.M[:,:,i1]
        H = B.H[:,:,i1]
        F = eigen(M)
        U = Diagonal(1 ./ sqrt.(F.values)) * F.vectors'
        Hnew = U * H * U'
        F = eigen(Hermitian(Hnew))
        B.spectrum[:,i1] = F.values
    end
    return A,B,Hn
end
