import Pkg; Pkg.activate(".")
using Revise
using Random
using SimulationCode

dims = 2
kBT = 0.1
ρ = 1.0
force_cutoff = 1.25             # Force cutoff
σ_ratio = 4.0                   # Ratio of the diameters of the smallest and largest particles
interaction_potential = SimulationCode.Berthier(
                                    n = 12,
                                    rc = force_cutoff,
                                    σ_ratio = σ_ratio,
                                    ζ = 0.2)

# System
Δt = 0.01                           # Time step
m = 1.0                             # mass

# system = Brownian(kBT, γ, Δt, dims)
system = SimulationCode.Newtonian(kBT, Δt, m, false, dims)
# system = Langevin(γ, kBT, Δt, m, dims)

N = 1200                        # Number of particles
q_cutoff = 0.0                  # Size of the shell of the calculation of the Steinhardt order parameters
N_stepsMC = 10^5                # Number of MC steps to take
N_stepsMD = 3*10^5                # Number of time steps to take
swap_probability = 0.1          # Probability of choosing swap over displacement
max_MC_displacement = 0.1       # Maximal displacement in a displacement step in one direction
N_MD_equilibration_steps = 10^3   # Number of steps for short-time MD equilibration



random_seed = rand(1:10^9)      # Seed for the random number generator
box_size = (N/ρ)^(1/dims)          # Cubic box dimension
simulation_folder = "Data"      # Name of folder in which to store datafile
simulation_name = joinpath(@__DIR__, simulation_folder, "T_$(kBT)_seed_$(random_seed)")    # Name of the datafile
simulation_suffix = "_Equilibration.h5"
simulation_name_full = simulation_name*simulation_suffix     # Name of the datafile

# For neighbor lists
skin_distanceMC = 0.6           # Size of the verlet cells for swap MC
skin_distanceMD = 0.4          # Size of the verlet cells for MD

dump_info = SimulationCode.DumpInfo(
    true, #save
    simulation_name_full,
    0:1000:N_stepsMD,#SimulationCode.create_when_to_save_array(N_stepsMD, 200), #when save
    true, #r
    true, #v
    true, #F
    false, #D
    false, #Epot      
)

cb(x...) = nothing 
# Initialize structs and set random seed
parameters = SimulationCode.ParameterStruct(
    N_MD_equilibration_steps, random_seed, ρ, N, box_size, N_stepsMC, swap_probability, max_MC_displacement, 
    force_cutoff^2, q_cutoff, system, interaction_potential, dump_info, cb)
Random.seed!(random_seed)

arrays = SimulationCode.ArrayStruct(N, dims)
SimulationCode.generate_diameters!(arrays, parameters, interaction_potential)

output = SimulationCode.OutputStruct() 
println("Random seed = $random_seed")
# Long Equilibration

neighborlist = SimulationCode.initialize_neighbor_struct(skin_distanceMC, box_size, force_cutoff, N, arrays.D_array, dims)
SimulationCode.find_random_initial_configuration!(arrays, parameters, output, neighborlist)

println("\nStarting Long Equilibriation Procedure\n\n")
SimulationCode.perform_swap_monte_carlo!(arrays, parameters, output, neighborlist)

parameters = SimulationCode.ParameterStruct(
    N_MD_equilibration_steps, random_seed, ρ, N, box_size, N_stepsMD, swap_probability, max_MC_displacement, 
    force_cutoff^2, q_cutoff, system, interaction_potential, dump_info, cb)

parameters.dump_info.filename = simulation_name*"_MD.h5"           # reset name of the datafile

neighborlist = SimulationCode.initialize_neighbor_struct(skin_distanceMD, box_size, force_cutoff, N, arrays.D_array, dims)

SimulationCode.perform_molecular_dynamics!(arrays, parameters, output, neighborlist)