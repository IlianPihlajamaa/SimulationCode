"""
    calculate_full_energy(arrays, parameters, neighborlist)

Calculates the potential energy of the system.

This function computes the total potential energy of a system of particles by 
iterating over each particle and summing the energy contributions from its 
interactions with neighboring particles. The energy is halved at the end to 
correct for double-counting each interaction.

# Parameters
- `arrays`: An object containing the arrays `r_array` and `D_array` which 
  represent the positions and other properties of the particles respectively.
- `parameters`: An object containing the parameters `N` (number of particles) 
  and `box_size` (size of the simulation box).
- `neighborlist`: A list or array that provides information about neighboring 
  particles for each particle in the system.

# Returns
- `Float64`: The total potential energy of the system, corrected for double-counting.
"""
function calculate_full_energy(arrays, parameters, neighborlist)
    energy = 0.0
    r_array = arrays.r_array
    D_array = arrays.D_array
    N = parameters.N
    box_size = parameters.box_size
    @inbounds for particle_i = 1:N
        ri = r_array[particle_i]
        Di = D_array[particle_i]
        energy += find_energy(particle_i, ri, Di, arrays, parameters, neighborlist)
    end
    return energy / 2.0 #double counting
end


"""
    find_kinetic_energy(arrays, parameters, output)

Calculates the kinetic energy of the system.

This function computes the total kinetic energy of a system of particles by 
summing the squared velocities of all particles and multiplying by half of 
the mass of the particles.

# Parameters
- `arrays`: An object containing the array `v_array` which represents the 
  velocities of the particles.
- `parameters`: An object containing the parameter `N` (number of particles) 
  and `system.m` (mass of the particles).
- `output`: An object where the calculated kinetic energy (`kinetic_energy`) 
  will be stored.

# Returns
- Nothing. The result is stored in the `output` object.
"""
function find_kinetic_energy(arrays, parameters, output)
    v_array = arrays.v_array
    Ekin = 0.0
    for i = 1:parameters.N
        Ekin += sum(v_array[i].^2)
    end
    Ekin *= parameters.system.m / 2.0
    output.kinetic_energy = Ekin
end

"""
    calculate_full_energy_no_neigh(arrays, parameters)

Calculates the potential energy of the system without using neighbor lists.

This function computes the total potential energy of a system of particles by 
iterating over each pair of particles and summing the energy contributions 
from their interactions. The energy is halved at the end to correct for 
double-counting each interaction.

# Parameters
- `arrays`: An object containing the arrays `r_array` and `D_array` which 
  represent the positions and other properties of the particles respectively.
- `parameters`: An object containing the parameters `N` (number of particles) 
  and `box_size` (size of the simulation box).

# Returns
- `Float64`: The total potential energy of the system, corrected for double-counting.
"""
function calculate_full_energy_no_neigh(arrays, parameters)
    interaction_potential = parameters.interaction_potential
    box_size = parameters.box_size
    r²_cutoff = parameters.force_cutoff2
    energy = 0.0
    N = parameters.N
    r_array = arrays.r_array
    D_array = arrays.D_array
    for particle_i = 1:N
        ri = r_array[particle_i]
        Di = D_array[particle_i]
        for particle_j = 1:N
            if particle_i == particle_j
                continue
            end
            rj = r_array[particle_j]
            Dj = D_array[particle_j]
            rij2 = calculate_squared_distance(ri,rj, box_size)
            mean_d = find_mean_D(Di, Dj, interaction_potential)
            mean_d_squared = mean_d * mean_d
            energy += ifelse(r²_cutoff * mean_d_squared < rij2, 0.0, potential(rij2, mean_d_squared, interaction_potential))
        end
    end
    return energy / 2.0
end

"""
    calculate_energy_no_neigh_i(particle_i, ri, Di, arrays, parameters)

Calculates the potential energy of a single particle due to its interactions 
with all other particles in the system without using neighbor lists.

This function computes the potential energy contribution of a single particle 
by iterating over all other particles and summing the energy contributions 
from their interactions.

# Parameters
- `particle_i`: Integer index of the particle for which the energy is calculated.
- `ri`: Position vector of particle `i`.
- `Di`: Property vector of particle `i` (e.g., diameter).
- `arrays`: An object containing the arrays `r_array` and `D_array` which 
  represent the positions and other properties of the particles respectively.
- `parameters`: An object containing the parameters including `N` (number of 
  particles), `interaction_potential`, `box_size` (size of the simulation box), 
  and `force_cutoff2` (squared cutoff distance for force calculations).

# Returns
- `Float64`: The potential energy of the specified particle due to its 
  interactions with all other particles.
"""
function calculate_energy_no_neigh_i(particle_i, ri, Di, arrays, parameters)
    interaction_potential = parameters.interaction_potential
    box_size = parameters.box_size
    r²_cutoff = parameters.force_cutoff2
    energy = 0.0
    N = parameters.N
    r_array = arrays.r_array
    D_array = arrays.D_array
    for particle_j = 1:N
        if particle_i == particle_j
            continue
        end
        rj = r_array[particle_j]
        Dj = D_array[particle_j]
        rij2 = calculate_squared_distance(ri, rj, box_size)
        mean_d = find_mean_D(Di, Dj, interaction_potential)
        mean_d_squared = mean_d * mean_d
        energy += ifelse(r²_cutoff * mean_d_squared < rij2, 0.0, potential(rij2, mean_d_squared, interaction_potential))
    end

    return energy 
end


"""
    compute_potential_energy!(arrays, parameters, output, neighborlist)

Calculates the potential energy of each particle and updates the potential 
energy array.

This function computes the potential energy for each particle in the system 
by iterating over all particles and summing the energy contributions from 
their interactions with neighboring particles. The energy for each particle 
is then stored in the `potential_energy_array` divided by 2 to correct for 
double-counting.

# Parameters
- `arrays`: An object containing the arrays `r_array` (positions), `D_array` 
  (properties), and `potential_energy_array` (to store the calculated potential 
  energies of the particles).
- `parameters`: An object containing the parameter `N` (number of particles).
- `output`: An object where the calculated data will be stored (though not 
  used directly in this function).
- `neighborlist`: A list or array that provides information about neighboring 
  particles for each particle in the system.

# Returns
- Nothing. The potential energy for each particle is stored in 
  `arrays.potential_energy_array`.
"""
function compute_potential_energy!(arrays, parameters, output, neighborlist)
    r_array = arrays.r_array
    D_array = arrays.D_array
    potential_energy_array = arrays.potential_energy_array
    for particle_i in 1:parameters.N
        ri = r_array[particle_i]
        Di = D_array[particle_i]
        Ei = find_energy(particle_i, ri, Di, arrays, parameters, neighborlist)
        potential_energy_array[particle_i] = Ei/2.0
    end
end

"""
    compute_potential_energy!(arrays, parameters, output, neighborlist, particle_list)

Calculates the potential energy of specified particles and updates the potential 
energy array.

This function computes the potential energy for each particle in the system 
specified by `particle_list` by iterating over them and summing the energy 
contributions from their interactions with neighboring particles. The energy 
for each particle is then stored in the `potential_energy_array` divided by 2 
to correct for double-counting.

# Parameters
- `arrays`: An object containing the arrays `r_array` (positions), `D_array` 
  (properties), and `potential_energy_array` (to store the calculated potential 
  energies of the particles).
- `parameters`: An object containing the parameter `N` (number of particles).
- `output`: An object where the calculated data will be stored (though not 
  used directly in this function).
- `neighborlist`: A list or array that provides information about neighboring 
  particles for each particle in the system.
- `particle_list`: A list of particle indices for which the potential energy 
  needs to be computed.

# Returns
- Nothing. The potential energy for each specified particle is stored in 
  `arrays.potential_energy_array`.
"""
function compute_potential_energy!(arrays, parameters, output, neighborlist, particle_list)
    r_array = arrays.r_array
    D_array = arrays.D_array
    potential_energy_array = arrays.potential_energy_array
    for particle_i in particle_list
        ri = r_array[particle_i]
        Di = D_array[particle_i]
        Ei = find_energy(particle_i, ri, Di, arrays, parameters, neighborlist)
        potential_energy_array[particle_i] = Ei/2.0
    end
end


"""
    find_volume_fraction(Di, box_size, dims)

Calculates the volume fraction occupied by particles in the simulation box.

This function computes the volume fraction occupied by particles in the 
simulation box based on the particle diameters (`Di`), size of the simulation 
box (`box_size`), and dimensionality of the system (`dims`).

# Parameters
- `Di`: Array of particle diameters.
- `box_size`: Size of the simulation box.
- `dims`: Dimensionality of the system.

# Returns
- `Float64`: The volume fraction occupied by particles in the simulation box.
"""
function find_volume_fraction(Di, box_size, dims)
    # @assert dims == 3
    Vbox = box_size^dims
    if dims == 1
        Vparticles = 0.0
        for D in Di
            Vparticles += D
        end
        return Vparticles / Vbox
    elseif dims == 2
        Vparticles = 0.0
        for D in Di
            Vparticles += π * D ^ 2 / 4
        end
        return Vparticles / Vbox
    elseif dims == 3
        Vparticles = 0.0
        for D in Di
            Vparticles += π * D ^ 3 / 6
        end
        return Vparticles / Vbox
    elseif dims==4
        Vparticles = 0.0
        for D in Di
            Vparticles += π^2 * D ^ 4 / 32
        end
        return Vparticles / Vbox
    end
    error("This is dimension is not implemented")
end


"""
    calculate_squared_distance(ri, rj, box_size)

Calculates the squared distance between particles considering periodic boundary conditions.

This function computes the squared distance between two particles `ri` and `rj` 
taking into account periodic boundary conditions defined by the simulation 
box size `box_size`.

# Parameters
- `ri`: Position vector of particle i.
- `rj`: Position vector of particle j.
- `box_size`: Size of the simulation box.

# Returns
- `Float64`: The squared distance between the particles `ri` and `rj`.

# Notes
- Periodic boundary conditions are applied to ensure that the distance is 
  calculated correctly across periodic boundaries.
- The distance is squared to avoid unnecessary square root calculations, 
  which can improve computational efficiency.
"""
@inline function calculate_squared_distance(ri, rj, box_size)
    dr = ri - rj
    dr -= round.(dr / box_size) * box_size
    return sum(dr.^2)
end


"""
    GC_inner_loop(interaction_potential, r_array, ri, N_neighbors, particle_i, D_array, box_size, r²_cutoff, neighbor_list)

Calculates the potential energy contribution of a particle using the Gaussian Core potential 
within its neighborhood.

This function computes the potential energy contribution of a particle `particle_i` 
using the Gaussian Core (GC) potential within its neighborhood defined by the 
`neighbor_list`.

# Parameters
- `interaction_potential`: An object containing parameters of the interaction 
  potential including the well depth (ϵ) and the core energy (Uc).
- `r_array`: Array containing the positions of all particles.
- `ri`: Position vector of the particle `particle_i`.
- `N_neighbors`: Number of neighboring particles considered in the neighborhood.
- `particle_i`: Index of the particle for which the potential energy is calculated.
- `D_array`: Array containing the properties of all particles.
- `box_size`: Size of the simulation box.
- `r²_cutoff`: Squared cutoff distance for the potential.
- `neighbor_list`: List containing the indices of neighboring particles for each 
  particle in the system.

# Returns
- `Float64`: The potential energy contribution of the particle `particle_i` 
  within its neighborhood.

# Notes
- This function calculates the potential energy contribution of a particle 
  considering the Gaussian Core (GC) potential and applying periodic boundary 
  conditions to account for particles across periodic boundaries.
- The potential energy is calculated based on the distances between particles 
  within the defined neighborhood.
- The `@turbo` macro is used for loop optimization, enhancing computational 
  efficiency.
"""

function GC_inner_loop(interaction_potential, r_array, ri, N_neighbors, particle_i, D_array, box_size, r²_cutoff, neighbor_list)
    ϵ = interaction_potential.ϵ
    Uc = interaction_potential.Uc
    xi = ri[1]
    yi = ri[2]
    zi = ri[3]
    r_array_r = reinterpret(reshape, Float64, r_array)
    energy = 0.0
    @turbo for neighbor_index = 1:N_neighbors
        particle_j = neighbor_list[neighbor_index, particle_i]
        xj = r_array_r[1, particle_j]
        yj = r_array_r[2, particle_j]
        zj = r_array_r[3, particle_j]
        dx = xi - xj
        dy = yi - yj
        dz = zi - zj
        dx -= round(dx / box_size) * box_size
        dy -= round(dy / box_size) * box_size
        dz -= round(dz / box_size) * box_size
        rij2 = dx^2 + dy^2 + dz^2
        invxi2 = rij2
        energy += ifelse(r²_cutoff < rij2, 0.0, ϵ*exp(-invxi2) - Uc)
    end
    return energy
end

function berthier_inner_loop3(interaction_potential, r_array, ri, N_neighbors, particle_i, D_array, box_size, r²_cutoff, neighbor_list)
    ζ = interaction_potential.ζ
    c0 = interaction_potential.c0
    c2 = interaction_potential.c2
    c4 = interaction_potential.c4
    xi = ri[1]
    yi = ri[2]
    zi = ri[3]
    Di = D_array[particle_i]
    r_array_r = reinterpret(reshape, Float64, r_array)
    energy = 0.0
    @turbo for neighbor_index = 1:N_neighbors
        particle_j = neighbor_list[neighbor_index, particle_i]
        xj = r_array_r[1, particle_j]
        yj = r_array_r[2, particle_j]
        zj = r_array_r[3, particle_j]
        Dj = D_array[particle_j]
        dx = xi - xj
        dy = yi - yj
        dz = zi - zj
        dx -= round(dx / box_size) * box_size
        dy -= round(dy / box_size) * box_size
        dz -= round(dz / box_size) * box_size
        rij2 = dx^2 + dy^2 + dz^2
        mean_d = (Di + Dj)*0.5 * (1.0 - ζ * abs(Di - Dj)) 
        mean_d_squared = mean_d * mean_d
        invxi2 = rij2/mean_d_squared#
        invxi4 = invxi2*invxi2#
        xi12 = 1.0/(invxi4*invxi4*invxi4)#
        energy += ifelse(r²_cutoff * mean_d_squared < rij2, 0.0, xi12 + c4 * invxi4 + c2*invxi2 + c0)
    end
    return energy
end


function berthier_inner_loop2(interaction_potential, r_array, ri, N_neighbors, particle_i, D_array, box_size, r²_cutoff, neighbor_list)
    ζ = interaction_potential.ζ
    c0 = interaction_potential.c0
    c2 = interaction_potential.c2
    c4 = interaction_potential.c4
    xi = ri[1]
    yi = ri[2]
    Di = D_array[particle_i]
    r_array_r = reinterpret(reshape, Float64, r_array)
    energy = 0.0
    @turbo for neighbor_index = 1:N_neighbors
        particle_j = neighbor_list[neighbor_index, particle_i]
        xj = r_array_r[1, particle_j]
        yj = r_array_r[2, particle_j]
        Dj = D_array[particle_j]
        dx = xi - xj
        dy = yi - yj
        dx -= round(dx / box_size) * box_size
        dy -= round(dy / box_size) * box_size
        rij2 = dx^2 + dy^2 
        mean_d = (Di + Dj)*0.5 * (1.0 - ζ * abs(Di - Dj)) 
        mean_d_squared = mean_d * mean_d
        invxi2 = rij2/mean_d_squared#
        invxi4 = invxi2*invxi2#
        xi12 = 1.0/(invxi4*invxi4*invxi4)#
        energy += ifelse(r²_cutoff * mean_d_squared < rij2, 0.0, xi12 + c4 * invxi4 + c2*invxi2 + c0)
    end
    return energy
end

function IPL_inner_loop(interaction_potential, r_array, ri, N_neighbors, particle_i, D_array, box_size, r²_cutoff, neighbor_list)
    ϵ = interaction_potential.ϵ
    σ = interaction_potential.σ
    n = interaction_potential.n
    Uc = interaction_potential.Uc
    xi = ri[1]
    yi = ri[2]
    zi = ri[3]
    r_array_r = reinterpret(reshape, Float64, r_array)
    energy = 0.0
    @turbo for neighbor_index = 1:N_neighbors
        particle_j = neighbor_list[neighbor_index, particle_i]
        xj = r_array_r[1, particle_j]
        yj = r_array_r[2, particle_j]
        zj = r_array_r[3, particle_j]
        dx = xi - xj
        dy = yi - yj
        dz = zi - zj
        dx -= round(dx / box_size) * box_size
        dy -= round(dy / box_size) * box_size
        dz -= round(dz / box_size) * box_size
        rij2 = dx^2 + dy^2 + dz^2
        mean_d = σ 
        mean_d_squared = mean_d * mean_d
        xi2 = mean_d_squared/rij2#
        xin = xi2^(n/2)
        energy += ifelse(r²_cutoff * mean_d_squared < rij2, 0.0, ϵ*xin-Uc)
    end
    return energy
end

function LJ_inner_loop(interaction_potential, r_array, ri, N_neighbors, particle_i, D_array, box_size, r²_cutoff, neighbor_list)
    ϵ = interaction_potential.ϵ
    σ = interaction_potential.σ
    Uc = interaction_potential.Uc
    xi = ri[1]
    yi = ri[2]
    zi = ri[3]
    r_array_r = reinterpret(reshape, Float64, r_array)
    energy = 0.0
    @turbo for neighbor_index = 1:N_neighbors
        particle_j = neighbor_list[neighbor_index, particle_i]
        xj = r_array_r[1, particle_j]
        yj = r_array_r[2, particle_j]
        zj = r_array_r[3, particle_j]
        Dj = D_array[particle_j]
        dx = xi - xj
        dy = yi - yj
        dz = zi - zj
        dx -= round(dx / box_size) * box_size
        dy -= round(dy / box_size) * box_size
        dz -= round(dz / box_size) * box_size
        rij2 = dx^2 + dy^2 + dz^2
        mean_d = σ 
        mean_d_squared = mean_d * mean_d
        xi2 = mean_d_squared/rij2#
        xi6 = xi2*xi2*xi2
        xi12 = xi6*xi6
        energy += ifelse(r²_cutoff * mean_d_squared < rij2, 0.0, 4ϵ*(xi12-xi6)-Uc)
    end
    return energy
end


"""
    find_energy(particle_i, ri, Di, arrays, parameters, neighborlist)

Calculates the potential energy of a single particle.

This function computes the potential energy of a single particle `particle_i` 
based on its position `ri`, properties `Di`, and interactions with neighboring 
particles defined by `neighborlist`.

# Parameters
- `particle_i`: Index of the particle for which the potential energy is calculated.
- `ri`: Position vector of the particle `particle_i`.
- `Di`: Property vector of the particle `particle_i`.
- `arrays`: An object containing the arrays `r_array` and `D_array` which 
  represent the positions and properties of the particles respectively.
- `parameters`: An object containing parameters including `box_size`, 
  `force_cutoff2`, `system.dims`, and `interaction_potential`.
- `neighborlist`: An object containing information about neighboring particles 
  for each particle in the system.

# Returns
- `Float64`: The potential energy of the particle `particle_i`.

# Notes
- The function determines the appropriate inner loop function to use based 
  on the dimensionality of the system and the type of interaction potential.
- If the number of neighbors of the particle `particle_i` is 0, the function 
  returns 0.0.
- If the interaction potential is of type `Berthier`, `GaussianCore`, `LJ`, 
  or `PowerLaw`, specialized inner loop functions are called for improved 
  computational efficiency.
- Otherwise, a generic inner loop is used to calculate the potential energy 
  considering all neighboring particles within the cutoff distance.
- The potential energy is calculated based on the distances between particles 
  and their properties.
"""
function find_energy(particle_i, ri, Di, arrays, parameters, neighborlist)
    """Calculates the potential energy of a single particle"""
    box_size = parameters.box_size
    interaction_potential = parameters.interaction_potential
    r²_cutoff = parameters.force_cutoff2
    energy = 0.0
    N_neighbors = neighborlist.neighbor_numbers_full[particle_i]
    if N_neighbors == 0
        return 0.0
    end

    r_array = arrays.r_array
    D_array = arrays.D_array
    neighbor_list = neighborlist.neighbor_list_full
    if parameters.system.dims == 3 && typeof(interaction_potential)==Berthier
        return berthier_inner_loop3(interaction_potential, r_array, ri, N_neighbors, particle_i, D_array, box_size, r²_cutoff, neighbor_list)
    elseif parameters.system.dims == 2 && typeof(interaction_potential)==Berthier
        return berthier_inner_loop2(interaction_potential, r_array, ri, N_neighbors, particle_i, D_array, box_size, r²_cutoff, neighbor_list)
    elseif parameters.system.dims == 3 && typeof(interaction_potential)==GaussianCore
        return GC_inner_loop(interaction_potential, r_array, ri, N_neighbors, particle_i, D_array, box_size, r²_cutoff, neighbor_list)
    elseif parameters.system.dims == 3 && typeof(interaction_potential)==LJ
        return LJ_inner_loop(interaction_potential, r_array, ri, N_neighbors, particle_i, D_array, box_size, r²_cutoff, neighbor_list)
    elseif parameters.system.dims == 3 && typeof(interaction_potential)==PowerLaw
        return IPL_inner_loop(interaction_potential, r_array, ri, N_neighbors, particle_i, D_array, box_size, r²_cutoff, neighbor_list)
    else
        @inbounds for neighbor_index = 1:N_neighbors
            particle_j = neighbor_list[neighbor_index, particle_i]
            rj = r_array[particle_j]
            Dj = D_array[particle_j]
            rij2 = calculate_squared_distance(ri, rj, box_size)
            mean_d = find_mean_D(Di, Dj, interaction_potential)
            mean_d_squared = mean_d * mean_d
            if r²_cutoff * mean_d_squared > rij2
                energy += potential(rij2, mean_d_squared, interaction_potential)
            end
        end
    end
    return energy
end

"""
    find_pair_energy(ri, rj, Dj, box_size, parameters)

Calculates the potential energy between a pair of particles.

This function computes the potential energy between a pair of particles 
specified by their positions `ri` and `rj`, properties `Dj`, and the simulation 
box size `box_size`.

# Parameters
- `ri`: Position vector of the first particle.
- `rj`: Position vector of the second particle.
- `Dj`: Property vector of the second particle (e.g., diameter).
- `box_size`: Size of the simulation box.
- `parameters`: An object containing parameters including `force_cutoff2` and 
  `interaction_potential`.

# Returns
- `Float64`: The potential energy between the pair of particles.

# Notes
- The potential energy is calculated based on the squared distance between 
  the particles `ri` and `rj` considering periodic boundary conditions.
- The mean diameter `mean_d` between the particles is computed based on their 
  properties and the interaction potential.
- The potential energy is evaluated using the interaction potential and 
  corrected to avoid double-counting if the squared distance exceeds the 
  cutoff distance.
"""
function find_pair_energy(ri,rj, Dj, box_size, parameters)
    r²_cutoff = parameters.force_cutoff2
    interaction_potential = parameters.interaction_potential
    rij2 = calculate_squared_distance(ri,rj, box_size)
    mean_d = find_mean_D(Di, Dj, interaction_potential)
    mean_d_squared = mean_d * mean_d
    energy += ifelse(r²_cutoff * mean_d_squared < rij2, 0.0, potential(rij2, mean_d_squared, interaction_potential))
    return energy
end


"""
    populate_random_displacement_array!(a, N, max_displacement)

Populates a displacement array for a translation Monte Carlo step.

This function fills the array `a` with random displacements within the range 
[-`max_displacement`, `max_displacement`] for performing translation Monte Carlo 
steps.

# Parameters
- `a`: Array to be populated with random displacements.
- `N`: Number of elements in the array.
- `max_displacement`: Maximum displacement value.

# Returns
- Nothing. The array `a` is populated in-place with random displacements.

# Notes
- The function utilizes the `rand!` function to generate random numbers in the 
  range [0, 1] and then scales and shifts them to the desired range.
- The displacements are random and independent for each element in the array.
"""

@inline function populate_random_displacement_array!(a, N, max_displacement)
    rand!(a)
    for j = 1:N
        a[j] = a[j] * 2 * max_displacement .- max_displacement
    end
end

"""
    logdata(t::Int)

Checks if the given integer `t` is a power of 10 or a multiple of a power of 10.

This function evaluates whether the integer `t` is a power of 10 or a multiple 
of a power of 10, and returns `true` if it meets either condition, and `false` otherwise.

# Parameters
- `t::Int`: The integer to be checked.

# Returns
- `Bool`: `true` if `t` is a power of 10 or a multiple of a power of 10, `false` otherwise.

# Notes
- If `t` is equal to 0, the function returns `true` to handle the special case.
- For other positive integers `t`, the function checks if the logarithm base 10 
  of `t` is an integer and if `t` is divisible by the corresponding power of 10.
- The function returns `false` for negative integers.
"""
function logdata(t::Int)
    if t == 0
        return true
    end
    log10t = log10(t)
    if log10t >= 2 && t % 10^floor(Int64, log10t) == 0
        return true
    end
    return false
end

"""
    calculate_ql(arrays, parameters, output, neighborlist, l)

Calculates the non-averaged Steinhardt order parameters of order `l`.

This function computes the non-averaged Steinhardt order parameters of order `l` 
for each particle in the system based on its neighboring particles defined by 
`neighborlist`. The calculated order parameters are stored in the `output` object.

# Parameters
- `arrays`: An object containing the arrays `r_array` and `D_array` which 
  represent the positions and properties of the particles respectively.
- `parameters`: An object containing parameters including `N` (number of 
  particles) and `system.dims`.
- `output`: An object where the calculated order parameters will be stored.
- `neighborlist`: An object containing information about neighboring particles 
  for each particle in the system.
- `l`: Order of the Steinhardt order parameter.

# Returns
- Nothing. The calculated order parameters are stored in the `output` object.

# Notes
- This function assumes that the system dimensionality is 3.
- The non-averaged Steinhardt order parameter `q6_total` is computed for each 
  particle using the formula:
    q6_total(i) = sqrt(4π / (2l+1) * sum_m(|q6m(i)|^2))
  where:
    - `q6m(i)` is the m-th component of the Steinhardt order parameter for 
      particle `i`.
    - `Nneigh` is the number of neighboring particles of particle `i`.
    - `rij` is the distance vector between particles `i` and `j`.
    - `Y6m(rij)` is the spherical harmonics of order `l` and degree `m`.
- The resulting order parameters are averaged over all particles and stored in 
  the `output` object.
- If the order `l` is equal to 6, the calculated order parameter is stored in 
  `output.q6`; if `l` is equal to 4, it is stored in `output.q4`.
- For other orders, the function prints the result to the console.
"""
function calculate_ql(arrays, parameters, output, neighborlist, l)
    @assert parameters.system.dims == 3

    """
    Calculates the non-averaged Steinhardt order parameters
    """
    #q6_total(i) = sqrt(4pi/(2l+1)  sum_m |q6m(i)|^2)
    #q6m(i) = 1/Nneigh sum_j=neigh Y6m(rij)
    q6_total_averaged = 0.0
    for particle_i = 1:parameters.N
        q6_total = 0.0
        for m = -l:l
            q6_total += abs(calculate_qlm(arrays, parameters, neighborlist, particle_i, l, m))^2
        end
        q6_total = sqrt(4π / (2l + 1) * q6_total)
        q6_total_averaged += q6_total
    end
    q6_total_averaged /= parameters.N
    if l == 6
        output.q6 = q6_total_averaged
    elseif l == 4
        output.q4 = q6_total_averaged
    else
        println(q6_total_averaged)
    end
end

"""
    calculate_qlm(arrays, parameters, neighborlist, particle_i, l, m)

Calculates the qlm value for a specified particle.

This function computes the qlm value for a specified particle `particle_i` 
based on its neighboring particles defined by `neighborlist`, with respect 
to the specified spherical harmonics order `l` and degree `m`.

# Parameters
- `arrays`: An object containing the array `r_array` which represents the 
  positions of the particles.
- `parameters`: An object containing parameters including `system.dims`, 
  `box_size`, and `q_cutoff`.
- `neighborlist`: An object containing information about neighboring particles 
  for each particle in the system.
- `particle_i`: Index of the particle for which the qlm value is calculated.
- `l`: Order of the spherical harmonics.
- `m`: Degree of the spherical harmonics.

# Returns
- Tuple{Int, Complex}: A tuple containing the number of neighboring particles 
  within the cutoff distance `N_neighborsq`, and the calculated qlm value.

# Notes
- This function assumes that the system dimensionality is 3.
- The qlm value is computed for the specified particle `particle_i` by summing 
  the spherical harmonics Y6m(rij) over all neighboring particles within the 
  cutoff distance, and then dividing by the total number of such neighboring 
  particles `N_neighborsq`.
- The distance rij between particles `i` and `j` is computed considering 
  periodic boundary conditions.
- The cutoff distance q_cutoff is used to exclude particles that are too far 
  from the reference particle.
- If no neighboring particles are found within the cutoff distance, the function 
  returns `(0, 0.0 + 0.0im)` indicating zero neighboring particles and a qlm 
  value of zero.
"""
function calculate_qlm(arrays, parameters, neighborlist, particle_i, l, m)
    @assert parameters.system.dims == 3

    """
    Calculates qlm = 1/Nneigh sum_j=neigh Y6m(rij) for particle_i
    """
    r_array = arrays.r_array
    box_size = parameters.box_size
    xi = r_array[particle_i][1]
    yi = r_array[particle_i][2]
    zi = r_array[particle_i][3]
    N_neighbors = neighborlist.neighbor_numbers_full[particle_i]
    neighbor_list = neighborlist.neighbor_list_full
    q_cutoff2 = parameters.q_cutoff^2
    q6m = 0.0 + 0.0im
    Y6m = 0.0 + 0.0im
    N_neighborsq = 0
    @inbounds for neighbor_index = 1:N_neighbors
        particle_j = neighbor_list[neighbor_index, particle_i]
        xj = r_array[particle_j][1]
        yj = r_array[particle_j][2]
        zj = r_array[particle_j][3]
        xij = xi - xj
        yij = yi - yj
        zij = zi - zj
        xij -= round(xij / box_size) * box_size
        yij -= round(yij / box_size) * box_size
        zij -= round(zij / box_size) * box_size
        if xij^2 + yij^2 + zij^2 > q_cutoff2
            continue
        end
        N_neighborsq += 1
        @fastmath theta = atan(sqrt(xij^2 + yij^2) / zij)
        if xij == 0.0
            xij += 10^-10
        end

        @fastmath phi = atan(yij / xij)

        Y6m = SphericalHarmonics.sphericalharmonic(theta, phi, l=l, m=m)
        q6m += Y6m
    end
    if N_neighborsq == 0
        return N_neighborsq, q6m
    else
        q6m /= N_neighborsq
        return N_neighborsq, q6m
    end
end


"""
    calculate_ql_averaged(arrays, parameters, output, neighborlist, l)

Calculates the averaged Steinhardt order parameters of order `l`.

This function computes the averaged Steinhardt order parameters of order `l` 
for each particle in the system based on its neighboring particles defined by 
`neighborlist`. The calculated order parameters are stored in the `output` object.

# Parameters
- `arrays`: An object containing the arrays `r_array` and `D_array` which 
  represent the positions and properties of the particles respectively.
- `parameters`: An object containing parameters including `N` (number of 
  particles), `system.dims`, `box_size`, and `q_cutoff`.
- `output`: An object where the calculated order parameters will be stored.
- `neighborlist`: An object containing information about neighboring particles 
  for each particle in the system.
- `l`: Order of the Steinhardt order parameter.

# Returns
- Nothing. The calculated order parameters are stored in the `output` object.

# Notes
- This function assumes that the system dimensionality is 3.
- The averaged Steinhardt order parameter `q6_total_averaged` is computed for 
  each particle using the formula:
    q6_total_averaged(i) = sqrt(4π / (2l+1) * sum_m(|q6_totalk|^2))
  where:
    - `q6_totalk` is the qlm value for particle `i` and degree `m`.
    - `N_neighborsq` is the number of neighboring particles within the cutoff 
      distance for particle `i`.
    - The summation is performed over all degrees `m` from `-l` to `l`.
- The resulting order parameters are averaged over all particles and stored in 
  the `output` object.
- If the order `l` is equal to 6, the calculated order parameter is stored in 
  `output.q6`; if `l` is equal to 4, it is stored in `output.q4`.
- For other orders, the function prints the result to the console.
"""
function calculate_ql_averaged(arrays, parameters, output, neighborlist, l)
    @assert parameters.system.dims == 3

    """
    Calculates the averaged Steinhardt order parameters
    """
    qlm_array = zeros(Complex{Float64}, 2l + 1, parameters.N)
    N_neighbors_array = zeros(Int64, 2l + 1, parameters.N)
    r_array = arrays.r_array
    box_size = parameters.box_size
    q_cutoff2 = parameters.q_cutoff^2
    q6_total_averaged = 0.0
    for particle_i = 1:parameters.N
        for m = -l:l
            N_Neighbors, q6_totalk = calculate_qlm(arrays, parameters, neighborlist, particle_i, l, m)
            qlm_array[m+l+1, particle_i] = q6_totalk
            N_neighbors_array[m+l+1, particle_i] = N_Neighbors
        end
    end

    for particle_i = 1:parameters.N
        xi = r_array[particle_i][1]
        yi = r_array[particle_i][2]
        zi = r_array[particle_i][3]
        q6_total = 0.0
        for m = -l:l
            N_neighbors = neighborlist.neighbor_numbers_full[particle_i]
            neighbor_list = neighborlist.neighbor_list_full
            q6_totalk = qlm_array[m+l+1, particle_i]
            for neighbor_index = 1:N_neighbors
                particle_j = neighbor_list[neighbor_index, particle_i]
                xj = r_array[particle_j][1]
                yj = r_array[particle_j][2]
                zj = r_array[particle_j][3]
                xij = xi - xj
                yij = yi - yj
                zij = zi - zj
                xij -= round(xij / box_size) * box_size
                yij -= round(yij / box_size) * box_size
                zij -= round(zij / box_size) * box_size
                if xij^2 + yij^2 + zij^2 > q_cutoff2
                    continue
                end
                q6_totalk += qlm_array[m+l+1, particle_j]
            end
            N_neighborsq = N_neighbors_array[m+l+1, particle_i]
            q6_totalk /= (N_neighborsq + 1)
            q6_total += abs(q6_totalk)^2
        end
        q6_total = sqrt(q6_total * 4π / (2l + 1))
        q6_total_averaged += q6_total
    end

    q6_total_averaged /= parameters.N
    if l == 6
        output.q6 = q6_total_averaged
    elseif l == 4
        output.q4 = q6_total_averaged
    else
        println(q6_total_averaged)
    end
end

"""
    find_exponential_time_array(stop, factor)

Returns an array of exponentially spaced integers.

This function generates an array of exponentially spaced integers starting 
from 0 up to but not including `stop`, with each subsequent element in the 
array being `factor` times the previous element.

# Parameters
- `stop`: The upper limit (exclusive) for the array.
- `factor`: The factor by which each subsequent element is multiplied to 
  obtain the next element.

# Returns
- `Vector{Int64}`: An array of exponentially spaced integers.

# Notes
- The first element of the array is always 0.
- The function continues adding elements to the array until the next element 
  would exceed `stop`.
- The elements in the array are calculated as powers of `factor`, starting 
  from 0.
"""
function find_exponential_time_array(stop, factor)
    """
    returns an array of exponentially spaced integers
    """
    t_arr = Int64[]
    push!(t_arr, 0)
    t = 1
    while t < stop
        push!(t_arr, t)
        t = ceil(Int64, t * factor)
    end
    return t_arr
end

"""
    calculate_F2_self!(arrays, parameters, output)

Calculates the self part of the structure factor F2.

This function computes the self part of the scattering function F2 for a system 
of particles using their positions stored in `arrays.r_old_array` and `arrays.r_array`. 
The calculated structure factor is stored in the `output` object.

# Parameters
- `arrays`: An object containing arrays `r_old_array` and `r_array`, which 
  represent the previous and current positions of the particles respectively.
- `parameters`: An object containing parameters relevant to the calculation.
- `output`: An object where the calculated structure factor will be stored.

# Returns
- Nothing. The calculated structure factor is stored in the `output` object.

# Notes
- The self part of the structure factor F2 is calculated for a system of particles 
  based on their positions.
- The structure factor is computed for a given wave vector `kx`.
- The calculation involves summing contributions from each particle's position.
- The resulting structure factor is averaged over all particles and stored in 
  the `output` object as `F2s`.
"""
function calculate_F2_self!(arrays, parameters, output)

    rold = arrays.r_old_array
    r = arrays.r_array
    kx = 7.2

    N = length(rold)
    F2s = 0.0
    for i = 1:N
        rx = rold[i][1]
        krold = kx * rx 
        sinkrold, coskrold = sincos(krold)
        rx = r[i][1]
        kr = kx * rx 
        sinkr, coskr = sincos(kr)
        F2s += coskrold * coskr + sinkrold * sinkr
    end
    F2s /= N
    output.F2s = F2s
end
