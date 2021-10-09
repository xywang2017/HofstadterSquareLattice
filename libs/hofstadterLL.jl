using ClassicalOrthogonalPolynomials

mutable struct HofstadterLL
    p::Int 
    q::Int 
    nLL::Int
    lB::Float64

    l1::Int
    l2::Int 
    kvec::Matrix{ComplexF64}
    k1::Vector{Float64}
    k2::Vector{Float64}

    ϵ0::Float64  # energy unit
    a::Float64
    m::Float64
    g0::Float64
    V0::Float64
    gvec::Matrix{ComplexF64}

    O0::Array{ComplexF64,4}
    O1::Array{ComplexF64,4}
    H::Array{ComplexF64,4}
    spectrum::Array{Float64,3}

    
    HofstadterLL() = new()
end

function constructLLHofstadter(;p::Int=1,q::Int=10,lk::Int=30,m::Float64=1.0,a::Float64=1.0,V0::Float64=3.0,nLL::Int=20)
    A = HofstadterLL()
    A.p = p
    A.q = q 
    A.l1 = lk 
    A.l2 = lk 
    A.m = m 
    A.a = a 
    A.V0 = V0 
    A.g0 = 2π/A.a 
    A.nLL = nLL
    A.lB = A.a / sqrt(2π * A.p/A.q)

    A.k1 = collect(0:(A.l1-1))/A.l1 
    A.k2 = collect(0:(A.l2-1))/A.l2 * A.p/A.q 
    A.kvec = reshape(A.k1,:,1)*A.g0 .+ 1im * reshape(A.k2,1,:)*A.g0 

    A.ϵ0 = (2π)^2/(2*A.m*A.a^2)

    # ---- below works for p = 1 ---- #
    A.O0 = zeros(ComplexF64,nLL,nLL,A.l1,A.l2)
    A.O1 = zeros(ComplexF64,nLL,nLL,A.l1,A.l2)

    # diagonal part
    for n in 0:(nLL-1)
        A.O0[n+1,n+1,:,:] .= 1/(A.m*A.lB^2) * (n+0.5) /A.ϵ0
    end

    x = 0.5*A.g0^2 * A.lB^2
    # non-diagonal part 
    tmp = zeros(ComplexF64,nLL,nLL)
    for n in 0:(nLL-1)
        for m in 0:(nLL-1)
            if (m >= n)
                tmp[n+1,m+1] = -A.V0/A.ϵ0 * sqrt(factorial(big(n))/factorial(big(m))) * laguerrel(n,m-n,x) * exp(-x/2)
            else
                tmp[n+1,m+1] = -A.V0/A.ϵ0 * sqrt(factorial(big(m))/factorial(big(n))) * laguerrel(m,n-m,x) * exp(-x/2)
            end
        end
    end
    
    for n in 0:(nLL-1)
        for m in 0:(nLL-1)
            if (m>=n)
                term = tmp[n+1,m+1]* (A.g0*A.lB/sqrt(2))^(m-n) * exp.(-1im * real(A.kvec) * A.q/A.p)  +
                        tmp[n+1,m+1]* (-A.g0*A.lB/sqrt(2))^(m-n) * exp.(1im * real(A.kvec) * A.q/A.p)  +
                        tmp[n+1,m+1]* (1im * A.g0*A.lB/sqrt(2))^(m-n) * exp.(1im * imag(A.kvec) * A.q/A.p) + 
                        tmp[n+1,m+1]* (-1im * A.g0*A.lB/sqrt(2))^(m-n) * exp.(- 1im * imag(A.kvec) * A.q/A.p)
            else
                term = tmp[n+1,m+1]* (-A.g0*A.lB/sqrt(2))^(n-m) * exp.(-1im * real(A.kvec) * A.q/A.p)  +
                        tmp[n+1,m+1]* (A.g0*A.lB/sqrt(2))^(n-m) * exp.(1im * real(A.kvec) * A.q/A.p)  +
                        tmp[n+1,m+1]* (1im * A.g0*A.lB/sqrt(2))^(n-m) * exp.(1im * imag(A.kvec) * A.q/A.p) + 
                        tmp[n+1,m+1]* (-1im * A.g0*A.lB/sqrt(2))^(n-m) * exp.(- 1im * imag(A.kvec) * A.q/A.p)
            end
            A.O1[n+1,m+1,:,:] = term
        end
    end

    A.H = A.O0 + A.O1 
    A.spectrum = zeros(Float64,nLL,A.l1,A.l2)
    for i2 in eachindex(A.k2)
        for i1 in eachindex(A.k1)
            H = view(A.H,:,:,i1,i2)
            if norm(H-H')>1e-6
                println("Error with Hermiticity")
            end
            A.spectrum[:,i1,i2] = eigvals(Hermitian(H))
        end
    end
    return A
end