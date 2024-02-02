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
    Langevin(γ, kBT, Δt, m, dims) = new(γ, kBT, Δt, m, false, dims)
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

# Struct that holds all parameters
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


    
# Struct for passing around all arrays easily
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


# Struct that holds the neighbor lists
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


# Struct for storing output data
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