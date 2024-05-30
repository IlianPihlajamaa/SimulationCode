# SimulationCode.jl
SimulationCode.jl is a Julia package designed for running molecular dynamics and Monte Carlo simulations. This package provides a framework for simulating the behavior of particles in various physical systems using different simulation methods.

## Features
Support for Multiple Simulation Methods: Includes Newtonian dynamics, Brownian dynamics, and Langevin dynamics.

## Installation
To install SimulationCode.jl, run:

```julia
Copy code
import Pkg
Pkg.activate(".")
Pkg.add("https://github.com/IlianPihlajamaa/SimulationCode")
```

## Usage
Below is a sample script to illustrate how to use SimulationCode.jl for setting up and running a simulation:

```julia
Copy code
import Pkg; Pkg.activate(".")
using Random
using SimulationCode

# Simulation parameters
dims = 2
kBT = 0.1
ρ = 1.0

interaction_potential = SimulationCode.Berthier(
    -1.924145348608,
    2.11106232532992, 
    -0.5910974510923776, 
    0.2, 
    2.219
) 

# System setup
Δt = 0.01  # Time step
m = 1.0    # Mass
system = SimulationCode.Newtonian(kBT, Δt, m, false, dims) # also supports Langevin and Brownian dynamics

N = 1200  # Number of particles
force_cutoff = 1.25
q_cutoff = 0.0
N_stepsMC = 10^5  # Number of Monte Carlo steps
N_stepsMD = 3*10^5  # Number of molecular dynamics steps
swap_probability = 0.1
max_MC_displacement = 0.1
N_MD_equilibration_steps = 10^3

random_seed = rand(1:10^9)
box_size = (N / ρ)^(1 / dims)
simulation_folder = "Data"
simulation_name = joinpath(@__DIR__, simulation_folder, "T_$(kBT)_seed_$(random_seed)")
simulation_suffix = "_Equilibration.h5"
simulation_name_full = simulation_name * simulation_suffix

# Neighbor list parameters
skin_distanceMC = 0.6
skin_distanceMD = 0.4

dump_info = SimulationCode.DumpInfo(
    true,  # Save data
    simulation_name_full,
    0:1000:N_stepsMD,
    true,  # Save positions
    true,  # Save velocities
    true,  # Save forces
    false,  # Don't save diameters
    false   # Don't save potential energy
)

cb(x...) = nothing

# Initialize parameters and set random seed
parameters = SimulationCode.ParameterStruct(
    N_MD_equilibration_steps, random_seed, ρ, N, box_size, N_stepsMC, swap_probability, max_MC_displacement,
    force_cutoff^2, q_cutoff, system, interaction_potential, dump_info, cb
)
Random.seed!(random_seed)

arrays = SimulationCode.ArrayStruct(N, dims)
SimulationCode.generate_diameters!(arrays, parameters, interaction_potential)

output = SimulationCode.OutputStruct()
println("Random seed = $random_seed")

# Equilibration
neighborlist = SimulationCode.initialize_neighbor_struct(skin_distanceMC, box_size, force_cutoff, N, arrays.D_array, dims)
SimulationCode.find_random_initial_configuration!(arrays, parameters, output, neighborlist)

println("\nStarting Long Equilibration Procedure\n\n")
SimulationCode.perform_swap_monte_carlo!(arrays, parameters, output, neighborlist)

# Update parameters for MD
parameters = SimulationCode.ParameterStruct(
    N_MD_equilibration_steps, random_seed, ρ, N, box_size, N_stepsMD, swap_probability, max_MC_displacement,
    force_cutoff^2, q_cutoff, system, interaction_potential, dump_info, cb
)
parameters.dump_info.filename = simulation_name * "_MD.h5"

neighborlist = SimulationCode.initialize_neighbor_struct(skin_distanceMD, box_size, force_cutoff, N, arrays.D_array, dims)

SimulationCode.perform_molecular_dynamics!(arrays, parameters, output, neighborlist)
```
