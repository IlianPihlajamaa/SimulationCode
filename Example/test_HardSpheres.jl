import Pkg; Pkg.activate(".")
using Revise
using Random
using StaticArrays, HDF5
include("src/SimulationCode.jl")

   ## Example Weysser Mixture
    # kBT = 1.0                    # Temperature
    # δ = 0.1                      # Non-additivity
    # interaction_potential = LJ(1.0, 1.0)
    # ϕ = 0.58
    # ρ = 1.2#6ϕ/(π*(1.0+δ^2))         # Density
    dims = 3
    ## Example Berthier Mixture
    kBT = 1.0 #parse(Float64, ARGS[1])
    ρ = 0.94
    interaction_potential = SimulationCode.HardSphere()

    # System
    Δt = 0.01                           # Time step
    m = 1.0                             # mass
    system = SimulationCode.Brownian(kBT, 0.0, 0.0, dims)
    # system = SimulationCode.Newtonian(kBT, Δt, m, false, dims)
    # system = Langevin(kBT, 20, Δt, 1.0, dims)

    N = 10000                       # Number of particles
    force_cutoff = 1.25             # Force cutoff
    q_cutoff = 0.0                  # Size of the shell of the calculation of the Steinhardt order parameters
    N_stepsMC = 10^5                # Number of MC steps to take
    N_stepsMD = 0                # Number of time steps to take
    swap_probability = 0.0          # Probability of choosing swap over displacement
    max_MC_displacement = 0.1       # Maximal displacement in a displacement step in one direction
    N_MD_equilibration_steps = 0   # Number of steps for short-time MD equilibration


    random_seed = 12#rand(1:10^9)      # Seed for the random number generator
    box_size = (N/ρ)^(1/dims)          # Cubic box dimension
    simulation_folder = "Data"      # Name of folder in which to store datafile
    simulation_name = joinpath(@__DIR__, simulation_folder, "HS$(dims)_ρ_$(ρ)_seed_$(random_seed)")    # Name of the datafile
    simulation_suffix = "_Equilibration.h5"
    simulation_name_full = simulation_name*simulation_suffix     # Name of the datafile

    # For neighbor lists
    skin_distanceMC = 0.7           # Size of the verlet cells for swap MC
    skin_distanceMD = 0.0          # Size of the verlet cells for MD

    dump_info = SimulationCode.DumpInfo(
        true, #save
        simulation_name_full,
        0:1000:N_stepsMC,#SimulationCode.create_when_to_save_array(N_stepsMD, 200), #when save
        true, #r
        false, #v
        false, #F
        false, #D
        false, #Epot      
    )

    function create_chemical_pot_and_cavity_callback(Nbins, rmax, last_step)
        expnegβΔE_list = Float64[]
        y_bin_edges = range(0.0, rmax, length = Nbins+1)
        y = zeros(Nbins)
        y_counts = zeros(Int, Nbins)
        function cb2(arrays, parameters, output, neighborlist)
            if output.steps_done % 1000 == 0
                box_size = parameters.box_size
                
                begin
                    ################### compute μ
                    ΣexpnegβΔE = 0.0
                    count = 0
                    n = 100*parameters.N
                    kBT = parameters.system.kBT
                    β = 1/kBT
                    for _ = 1:n÷10
                        count += 1
                        ri = rand(eltype(arrays.r_array))*box_size
                        Di = 1.0
                        ΔE = SimulationCode.calculate_energy_no_neigh_i(0, ri, Di, arrays, parameters)
                        ΣexpnegβΔE += exp(-β*ΔE)
                    end
                    expnegβΔE = ΣexpnegβΔE /= count
                    push!(expnegβΔE_list, expnegβΔE)

                    ######################## compute y(r)
                    drmax = y_bin_edges[end]
                    bin_size = y_bin_edges[2] - y_bin_edges[1]
                    for _ = 1:n
                        particle_i = rand(1:parameters.N)
                        dr = rand()*drmax
                        if length(arrays.r_array[1]) == 2
                            ϕ = rand()*2π
                            dr_vec = SVector{2, Float64}(dr*cos(ϕ), dr*sin(ϕ))
                        elseif length(arrays.r_array[1]) == 3
                            ϕ = rand()*2π
                            θ = rand()*π
                            dr_vec = SVector{3, Float64}(dr*cos(ϕ)*sin(θ), dr*sin(ϕ)*sin(θ), dr*cos(θ))
                        end
                        ri_old = arrays.r_array[particle_i]
                        ri_new = ri_old + dr_vec
                        ri_new -= floor.(ri_new/box_size)*box_size

                        arrays.r_array[particle_i] = ri_new # displace

                        Di = 1.0
                        E = SimulationCode.calculate_energy_no_neigh_i(particle_i, ri_new, Di, arrays, parameters)
                        expnegβE = exp(-β*E)
                        bin_index = ceil(Int, dr/bin_size)
                        if 0 < bin_index <= Nbins
                            y[bin_index] += expnegβE
                            y_counts[bin_index] += 1
                        end

                        arrays.r_array[particle_i] = ri_old # place back
                    end

                end 
            end
            if output.steps_done == last_step
                μ = -kBT*log(sum(expnegβΔE_list)/length(expnegβΔE_list))
                bin_centers = [(y_bin_edges[i+1] + y_bin_edges[i])/2 for i = 1:Nbins]
                y[y_counts .> 0] ./= y_counts[y_counts .> 0]
                y .*= exp(μ/kBT)

                f = h5open(parameters.dump_info.filename, "cw")
                f["miscellaneous"]["chemical_potential"] = μ
                f["miscellaneous"]["cavity_distribution_function"] = y
                f["miscellaneous"]["cavity_distribution_function_r"] = bin_centers
                close(f)
            end
        end
        return cb2
    end


    cb = create_chemical_pot_and_cavity_callback(100, 4.0, N_stepsMC)
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

    neighborlist = SimulationCode.initialize_neighbor_struct(skin_distanceMC, box_size, force_cutoff, N, arrays.D_array, dims; maxneighbours=1000)
    SimulationCode.find_random_initial_configuration!(arrays, parameters, output, neighborlist, steps=10^4)
    println("\nStarting Long Equilibriation Procedure\n\n")
    SimulationCode.perform_swap_monte_carlo!(arrays, parameters, output, neighborlist)

    using Plots
    f = h5open(parameters.dump_info.filename)    
    y = read(f["miscellaneous"]["cavity_distribution_function"])
    r = read(f["miscellaneous"]["cavity_distribution_function_r"])
    plot(r,log.(y)) |> display
    close(f)