abstract type InteractionPotential end

struct HardSphere <: InteractionPotential end

struct Weysser <: InteractionPotential
    ϵ::Float64
    δ::Float64
end

struct LJ <: InteractionPotential
    ϵ::Float64
    σ::Float64
    rc::Float64
    Uc::Float64
    function LJ(ϵ, σ, rc) 
        Uc = 4ϵ * ((σ/rc)^12 -  (σ/rc)^6)
        new(ϵ, σ, rc, Uc)
    end
end

struct PowerLaw <: InteractionPotential
    ϵ::Float64
    σ::Float64
    n::Int
    rc::Float64
    Uc::Float64
    function PowerLaw(ϵ, σ, n, rc) 
        Uc = ϵ * (σ/rc)^n 
        new(ϵ, σ, n,  rc, Uc)
    end
end

struct GaussianCore <: InteractionPotential
    ϵ::Float64
    σ::Float64
    rc::Float64
    Uc::Float64
    function GaussianCore(ϵ, σ, rc) 
        Uc = ϵ * exp(- (rc/σ)^2) 
        new(ϵ, σ, rc, Uc)
    end
end


struct Berthier <: InteractionPotential 
    c0::Float64
    c2::Float64
    c4::Float64
    ζ::Float64
    σ_ratio::Float64
    n::Int64
    rc::Float64
    function Berthier(; n=12, rc=1.25, σ_ratio=2.219, ζ=0.2) 
        @assert n == 12
        c0 = -(1/8)*(8 + 6n + n^2)*rc^-n
        c2 = 1/4*n*(4 + n)*rc^(-2 - n)
        c4 = -(1/8)*n*(2 + n)*rc^(-4 - n)
        new(c0, c2, c4, ζ, σ_ratio, n, rc)
    end
end

abstract type System end

mutable struct Brownian <: System 
    kBT::Float64
    γ::Float64
    Δt::Float64
    dims::Int
end

mutable struct Newtonian <: System 
    kBT::Float64
    Δt::Float64
    m::Float64
    rescale_temperature::Bool
    dims::Int
end

mutable struct Langevin <: System 
    kBT::Float64
    γ::Float64
    Δt::Float64
    m::Float64
    rescale_temperature::Bool ## This one is always False
    dims::Int
    Langevin(kBT, γ, Δt, m, dims) = new(kBT, γ, Δt, m, false, dims)
end


mutable struct DumpInfo{Iterable}
    save::Bool
    filename::String
    when_to_save_array::Iterable
    save_r::Bool
    save_v::Bool
    save_F::Bool
    save_D::Bool
    save_Epot::Bool
end

"""
    mutable struct ParameterStruct{U, I, CB, D}

A mutable struct representing parameters for a simulation.

This struct holds various parameters necessary for running a simulation, such 
as the number of equilibration steps, random seed, density, system size, box 
size, number of steps, swap probability, maximum Monte Carlo displacement, 
force cutoff distance squared, cutoff distance for Steinhardt order parameters, 
system type, interaction potential, information dumping settings, and callback 
function.

# Fields
- `N_MD_equilibration_steps::Int64`: Number of equilibration steps in molecular 
  dynamics simulations.
- `random_seed::Int64`: Random seed for reproducibility.
- `ρ::Float64`: Density of the system.
- `N::Int64`: Number of particles in the system.
- `box_size::Float64`: Size of the simulation box.
- `N_steps::Int64`: Total number of steps in the simulation.
- `swap_probability::Float64`: Probability of attempting a particle swap in 
  Monte Carlo simulations.
- `max_MC_displacement::Float64`: Maximum Monte Carlo displacement.
- `force_cutoff2::Float64`: Cutoff distance squared for forces in interaction 
  potentials.
- `q_cutoff::Float64`: Cutoff distance for Steinhardt order parameters.
- `system::I`: Type of system.
- `interaction_potential::U`: Type of interaction potential.
- `dump_info::D`: Information dumping settings.
- `callback::CB`: Callback function.

# Type Parameters
- `U`: Type parameter representing the type of interaction potential.
- `I`: Type parameter representing the type of system.
- `CB`: Type parameter representing the type of callback function.
- `D`: Type parameter representing the type of information dumping settings.
"""
mutable struct ParameterStruct{U, I, CB, D}
    N_MD_equilibration_steps::Int64
    random_seed::Int64
    ρ::Float64
    N::Int64
    box_size::Float64
    N_steps::Int64
    swap_probability::Float64
    max_MC_displacement::Float64
    force_cutoff2::Float64
    q_cutoff::Float64
    system::I
    interaction_potential::U
    dump_info::D
    callback::CB
end


    
"""
    struct ArrayStruct{dims}

A struct representing arrays storing particle data.

This struct holds arrays for storing various particle data, such as positions, 
velocities, forces, diameters, potential energies, and random displacements.

# Fields
- `r_array::Vector{SVector{dims, Float64}}`: Holds the positions of all 
  particles.
- `r_old_array::Vector{SVector{dims, Float64}}`: Holds the positions of all 
  particles at the start of the simulation run.
- `D_array::Array{Float64, 1}`: Holds the diameters of all particles.
- `v_array::Vector{SVector{dims, Float64}}`: Holds the velocities of all 
  particles.
- `F_array::Vector{SVector{dims, Float64}}`: Holds the forces acting on all 
  particles.
- `random_displacement::Vector{SVector{dims, Float64}}`: Holds random numbers 
  used for a Monte Carlo step.
- `potential_energy_array::Array{Float64, 1}`: Holds the potential energies 
  of all particles.

# Type Parameters
- `dims`: Type parameter representing the dimensionality of the system.
"""
struct ArrayStruct{dims}
    r_array::Vector{SVector{dims, Float64}}                   # holds the positions of all particles
    r_old_array::Vector{SVector{dims, Float64}}               # holds the positions of all particles at start of run
    D_array::Array{Float64,1}                   # holds the diameters of all particles 
    v_array::Vector{SVector{dims, Float64}}                   # holds the velocities of all particles 
    F_array::Vector{SVector{dims, Float64}}                   # holds the forces of all particles 
    random_displacement::Vector{SVector{dims, Float64}}       # holds random numbers for a monte carlo step
    potential_energy_array::Array{Float64,1}
end

ArrayStruct(N, dims) = ArrayStruct{dims}(
    [zeros(SVector{dims}) for _ in 1:N], 
    [zeros(SVector{dims}) for _ in 1:N], 
    zeros(N), 
    [zeros(SVector{dims}) for _ in 1:N], 
    [zeros(SVector{dims}) for _ in 1:N], 
    [zeros(SVector{dims}) for _ in 1:N],
    zeros(N), 
    )    


    """
    struct NeighborListStruct{dims, dims2, A, B, C}

A struct representing a neighbor list for particles in a simulation.

This struct holds various data structures used for managing neighbor lists 
in simulations, including meta-data arrays, neighbor lists, displacement arrays, 
and auxiliary structures.

# Fields
- `skin_distance::Float64`: Skin distance used to update the neighbor list.
- `cell_size::Float64`: Size of the cells used for cell lists.
- `N_cells::Int64`: Number of cells in the system.
- `verlet_sphere_size::Float64`: Size of the Verlet sphere for neighbor searching.
- `neighbor_numbers_full::Array{Int64, 1}`: Meta-data array holding the number 
  of neighbors for each particle in the full neighbor list.
- `neighbor_numbers_half::Array{Int64, 1}`: Meta-data array holding the number 
  of neighbors for each particle in the half neighbor list.
- `neighbor_list_full::Array{Int64, 2}`: Full neighbor list storing indices 
  of neighboring particles for each particle.
- `neighbor_list_half::Array{Int64, 2}`: Half neighbor list storing indices 
  of neighboring particles for each particle, with each pair represented only 
  once.
- `displacement_array::Vector{SVector{dims, Float64}}`: Array holding the 
  displacements of all particles since the last neighbor list was calculated.
- `displacement_squared_array::Array{Float64, 1}`: Array holding the squared 
  total displacement of all particles.
- `auxiliary_struct::A`: Auxiliary structure for additional data or methods 
  related to neighbor list management.
- `celllist::C`: Cell list structure for efficient neighbor searching.
- `box::B`: Box structure representing the simulation box.

# Type Parameters
- `dims`: Type parameter representing the dimensionality of the system.
- `dims2`: Type parameter representing an additional dimensionality if needed.
- `A`: Type parameter representing the type of auxiliary structure.
- `B`: Type parameter representing the type of box structure.
- `C`: Type parameter representing the type of cell list structure.
"""
struct NeighborListStruct{dims, dims2, A, B, C}
    skin_distance::Float64
    cell_size::Float64
    N_cells::Int64
    verlet_sphere_size::Float64
    neighbor_numbers_full::Array{Int64,1}               # holds meta-data on the neighbor list
    neighbor_numbers_half::Array{Int64,1}               # holds meta-data on the neighbor list
    neighbor_list_full::Array{Int64,2}               # neighbor list
    neighbor_list_half::Array{Int64,2}               # neighbor list, each pair only once
    displacement_array::Vector{SVector{dims, Float64}}       # holds the displacements of all particles since the last neighbor list was calculated
    displacement_squared_array::Array{Float64, 1}       # holds the squared total displacement of all particles
    auxilliary_struct::A
    celllist::C
    box::B
end


"""
    mutable struct OutputStruct

A mutable struct representing output data from a simulation.

This struct holds various output data generated during a simulation, including 
potential energy, kinetic energy, number of Monte Carlo steps performed, 
acceptance counts for translations and swaps, total counts for translations 
and swaps, number of neighbor list rebuilds, and various order parameters.

# Fields
- `potential_energy::Float64`: Total potential energy of all particles.
- `kinetic_energy::Float64`: Total kinetic energy of all particles.
- `steps_done::Int64`: Number of Monte Carlo steps done.
- `N_translations_accepted::Int64`: Number of Monte Carlo translation steps accepted.
- `N_swaps_accepted::Int64`: Number of Monte Carlo swap steps accepted.
- `N_translations_total::Int64`: Total number of Monte Carlo translation steps proposed.
- `N_swaps_total::Int64`: Total number of Monte Carlo swap steps proposed.
- `N_neighbor_list_rebuilds::Int64`: Number of times the neighbor lists are rebuilt.
- `q4::Float64`: Q4 Steinhardt order parameter.
- `q6::Float64`: Q6 Steinhardt order parameter.
- `F2s::Float64`: Self-intermediate structure factor.

# Constructor
- `OutputStruct() = new(...)`: Constructor method creating a new instance of 
  `OutputStruct` with initialized fields.

# Notes
- This struct is typically used to accumulate and store various output data 
  during a simulation run.
"""
mutable struct OutputStruct
    potential_energy::Float64                   # Total potential energy of all particles
    kinetic_energy::Float64                     # Total kinetic energy of all particles
    steps_done::Int64                           # Number of MC steps done
    N_translations_accepted::Int64              # Number of MC translation steps accepted
    N_swaps_accepted::Int64                     # Number of MC swap steps accepted
    N_translations_total::Int64                 # Number of MC translation steps proposed
    N_swaps_total::Int64                        # Number of MC swap steps proposed
    N_neighbor_list_rebuilds::Int64             # Number of times the neighbor lists are built
    q4::Float64                                 # Q4 steinhardt order parameter
    q6::Float64                                 # Q6 steinhardt order parameter
    F2s::Float64                                # Self-intermediate structure factor
    OutputStruct() = new(
        0.0, 
        0.0, 
        0, 
        0, 
        0,
        0, 
        0, 
        0, 
        0.0, 
        0.0,
        0.0)
end

"""
    initialize_neighbor_struct(skin_distance, box_size, force_cutoff, N, D_array, dims; maxneighbours=200)

Initializes a neighbor list structure for particles in a simulation.

This function initializes a neighbor list structure based on the provided 
parameters, including the skin distance, box size, force cutoff distance, 
number of particles, particle diameters, and dimensionality of the system.

# Arguments
- `skin_distance`: Skin distance used to update the neighbor list.
- `box_size`: Size of the simulation box.
- `force_cutoff`: Cutoff distance for forces in interaction potentials.
- `N`: Number of particles in the system.
- `D_array`: Array containing the diameters of all particles.
- `dims`: Dimensionality of the system.
- `maxneighbours=200`: Maximum number of neighbors per particle.

# Returns
- `NeighborListStruct`: Initialized neighbor list structure.

# Notes
- This function initializes a neighbor list structure suitable for simulations 
  in 2D or 3D dimensions.
- The neighbor list structure is created based on the Verlet neighbor list 
  algorithm.
- It allocates memory for neighbor lists, displacement arrays, and auxiliary 
  structures.
- For higher dimensions, the function raises an error indicating that it needs 
  implementation.

"""
function initialize_neighbor_struct(skin_distance, box_size, force_cutoff, N, D_array, dims; maxneighbours=200)
    if dims == 2 || dims == 3
        σ_max = maximum(D_array)
        verlet_sphere_size = skin_distance+force_cutoff*σ_max
        N_cells = floor(Int64, box_size/(verlet_sphere_size))     # Number of bins in every direction in the verlet list
        cell_size = box_size / N_cells                            # One-dimensional size of the cells
        max_neighbors = maxneighbours                             # This relates to the amount of memory is allocated for each neighbor list
        x = rand(dims, N)*box_size
        box = Box(ones(dims)*box_size, verlet_sphere_size)
        cl = CellList(x, box, parallel=false)
        aux = CellListMap.AuxThreaded(cl)
        return NeighborListStruct{dims, dims^2, typeof(aux), typeof(box), typeof(cl)}(
            skin_distance, 
            cell_size, 
            N_cells, 
            verlet_sphere_size, 
            zeros(Int64, N), 
            zeros(Int64, N), 
            zeros(Int64, max_neighbors, N), 
            zeros(Int64, max_neighbors, N), 
            [zeros(SVector{dims}) for _ in 1:N], 
            zeros(N),
            aux,
            cl,
            box
        )
    else
        error("To be implemented")
    end
end