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

    a::Float64
    m::Float64
    g0::Float64
    V0::Float64
    gvec::Matrix{ComplexF64}

    O0::Array{Float64,4}
    O1::Array{Float64,4}
    H::Array{Float64,4}
    spectrum::Array{Float64,3}
    
    HofstadterLL() = new()
end

function constructLLHofstadter(;p::Int=1,q::Int=10,lk::Int=20,m::Float64=1.0,a::Float64=1.0,V0::Float64=3.0,nLL::Int=20)
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

    # ---- below works for p = 1 ---- #
    A.O0 = zeros(Float64,nLL,nLL,A.l1,A.l2)
    A.O1 = zeros(Float64,nLL,nLL,A.l1,A.l2)

    # diagonal part
    for n in 0:(nLL-1)
        A.O0[n+1,n+1,:,:] .= 1/(A.m*A.lB^2) * (n+0.5) 
        A.O0[n+1,n+1,:,:] .-= 2*A.V0 * ( cos.(real(A.kvec)*A.q/A.p) + 
                        real(exp.(1im*A.g0*imag(A.kvec)*A.lB^2 .-A.g0^2*A.lB^2/4)) *laguerrel(n,0,0.5*A.g0^2*A.lB^2) )
    end
    # non-diagonal part 
    for n in 0:(nLL-2)
        for m in (n+1):(nLL-1)
            A.O1[n+1,m+1,:,:] = -2*A.V0 * sqrt(factorial(big(n))/factorial(big(m))) *  
                        real(exp.(1im*A.g0*imag(A.kvec)*A.lB^2 .-A.g0^2*A.lB^2/4) *(1im*A.g0*A.lB/sqrt(2))^(m-n) ) * laguerrel(n,m-n,0.5*A.g0^2*A.lB^2) 
        end
    end

    A.H = A.O0 + A.O1 
    A.spectrum = zeros(Float64,nLL,A.l1,A.l2)
    for i2 in eachindex(A.k2)
        for i1 in eachindex(A.k1)
            A.spectrum[:,i1,i2] = eigvals(Hermitian(A.H[:,:,i1,i2]))
        end
    end
    return A
end