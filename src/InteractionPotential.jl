@inline function find_mean_D(Di, Dj, U::Union{HardSphere, Weysser})
    return (Di + Dj)*0.5 
end

@inline function find_mean_D(Di, Dj, U::Union{LJ, PowerLaw, GaussianCore})
    return U.σ^2
end

@inline function find_mean_D(Di, Dj, U::Berthier)
    return (Di + Dj)*0.5 * (1.0 - U.ζ * abs(Di - Dj)) 
end

@inline function potential(r_squared, mean_d_squared, ::HardSphere)
    return ifelse(r_squared < mean_d_squared, Inf64, 0.0)
end

@inline function potential(r_squared, mean_d_squared, U::LJ)

    xi2 = mean_d_squared/r_squared
    xi4 = xi2*xi2
    xi6 = xi2*xi4
    xi12 = xi6*xi6

    return 4U.ϵ*(xi12 - xi6) - U.Uc
end

"""
Calculates the Lennard-Jones force between two particles.

# Arguments
- `r_squared`: The squared distance between the two particles.
- `mean_d_squared`: The squared mean diameter of the two particles.
- `U::LJ`: The Lennard-Jones potential parameters.

# Returns
- The force divided by the distance between the two particles (F/r).

"""
@inline function force(r_squared, mean_d_squared, U::LJ)
    inv_r_squared = 1.0/r_squared
    xi2 = mean_d_squared*inv_r_squared
    xi6 = xi2*xi2*xi2
    xi12 = xi6*xi6
    Fdivr = U.ϵ*(48xi12 - 24*xi6)*inv_r_squared
    return Fdivr
end


@inline function potential(r_squared, mean_d_squared, U::PowerLaw)
    xi2 = mean_d_squared/r_squared
    xin = xi2^(U.n/2)
    return U.ϵ*xin - U.Uc
end

@inline function potential(r_squared, mean_d_squared, U::GaussianCore)
    r2divd2 = r_squared/mean_d_squared
    return @fastmath U.ϵ*exp(-r2divd2) - U.Uc
end

@inline function potential(r_squared, mean_d_squared, U::Weysser)
    """
    U = ϵ (xi)^36, where we defined xi = dij/rij
    """
    xi2 = mean_d_squared/r_squared
    xi4 = xi2*xi2
    xi8 = xi4*xi4
    xi16 = xi8*xi8
    xi36 = xi16*xi16*xi4
    return U.ϵ*xi36
end

@inline function force(r_squared, mean_d_squared, U::Weysser)
    """
    -∇U = 36ϵ (xi)^36 r^-2 vec(r), where we defined xi = dij/rij
    this function returns the force without the multiplication with vec(r)
    """
    inv_r_squared = 1.0/r_squared
    xi2 = mean_d_squared*inv_r_squared
    xi4 = xi2*xi2
    xi8 = xi4*xi4
    xi16 = xi8*xi8
    xi36 = xi16*xi16*xi4
    return 36*U.ϵ*xi36*inv_r_squared
end


@inline function potential(r_squared, mean_d_squared, U::Berthier)
    """
    Soft repulsive pair potential
    """
    invxi2 = r_squared/mean_d_squared
    invxi4 = invxi2*invxi2
    xi12 = 1.0/(invxi4*invxi4*invxi4)
    return xi12 + U.c4 * invxi4 + U.c2*invxi2 + U.c0
end


@inline function force(r_squared, mean_d_squared, U::Berthier)
    """
    Soft repulsive pair potential force (before multiplication with pair vector)
    """
    inv_mean_d2 = 1.0 / mean_d_squared
    invxi2 = r_squared*inv_mean_d2
    invxi4 = invxi2*invxi2
    xi14 = 1.0/(invxi4*invxi4*invxi4*invxi2)
    return -2.0 * inv_mean_d2 * (U.c2 + 2.0*U.c4*invxi2 - 6.0*xi14)
end
    


# @inline function WCA_potential(r_squared, mean_d_squared)
#     """
#     WCA potential
#     """
#     xi2 = mean_d_squared/r_squared
#     xi6 = xi2*xi2*xi2
#     xi12 = xi6*xi6
#     return 4.0 * (xi12 - xi6) + 1
# end

# @inline function WCA_force(r_squared, mean_d_squared)
#     """
#     WCA force (before multiplication with pair vector)
#     """
#     xi2 = mean_d_squared/r_squared
#     xi6 = xi2*xi2*xi2
#     xi12 = xi6*xi6
#     return 4.0 * (12*xi12 - 6*xi6)/r_squared
# end