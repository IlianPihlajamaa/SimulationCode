import Pkg; Pkg.activate(".")

include("src/SimulationCode.jl")

using HDF5, StaticArrays

function read_restart_corentin(file, timestep)
    f = h5open(file) 
    attribs = attrs(f)
    r_array = read(f, "positions/$(timestep)")
    v_array = read(f, "velocities/$(timestep)")
    D_array = read(f, "diameters/diameters")
    Δt = attribs["Δt"]
    simulation_name = attribs["simulation_name"]
    ρ = attribs["ρ"]
    N = attribs["N"]
    box_size = attribs["box_size"]
    swap_probability = attribs["swap_probability"]
    max_MC_displacement = attribs["max_MC_displacement"]
    force_cutoff2 = attribs["force_cutoff2"]
    N_stepsMD = attribs["N_stepsMD"]
    q_cutoff = attribs["q_cutoff"]
    interaction_potential = "Berthier"
    c0 = attribs["c0"]
    c2 = attribs["c2"]
    c4 = attribs["c4"]
    dims = 3
    N_stepsMD = attribs["N_stepsMD"]
    N_stepsMC = attribs["N_stepsMC"]
    random_seed = attribs["random_seed"]

    kBT = attribs["kBT"]
    m = attribs["m"]
    rescale_temperature = attribs["rescale_temperature"]
    N_MD_equilibration_steps = attribs["N_MD_equilibration_steps"]
    ζ = attribs["ζ"]
    σ_ratio = attribs["σ_ratio"]

    system = "Newtonian"
    cb(x...) = nothing 


    close(f)
    arrays = SimulationCode.ArrayStruct{dims}(
        Vector(reinterpret(reshape, SVector{dims, Float64}, r_array)), 
        Vector(reinterpret(reshape, SVector{dims, Float64}, copy(r_array))),
        D_array,
        Vector(reinterpret(reshape, SVector{dims, Float64}, v_array)), 
        [zeros(SVector{dims}) for _ in 1:N], 
        [zeros(SVector{dims}) for _ in 1:N],
        zeros(N),
        )

    if split(interaction_potential, '.')[end] == "Berthier"
        interaction_potential = SimulationCode.Berthier(
        c0,
        c2, 
        c4, 
        ζ, 
        σ_ratio)
    end
    if split(system, '.')[end] == "Newtonian"
        system = SimulationCode.Newtonian(
        kBT,
        Δt,
        m,
        rescale_temperature,
        dims)
    end
    dump_info = SimulationCode.DumpInfo(
        true, #save
        simulation_name,
        Set(SimulationCode.create_when_to_save_array(N_stepsMD, 200)), #when save
        true, #r
        true, #v
        true, #F
        false, #D
        false, #Epot      
    )

    parameters = SimulationCode.ParameterStruct(
        N_MD_equilibration_steps,
        random_seed,
        ρ,
        N,
        box_size,
        N_stepsMD,
        swap_probability,
        max_MC_displacement,
        force_cutoff2,
        q_cutoff,
        system,
        interaction_potential,
        dump_info,
        cb
    )
    output = SimulationCode.OutputStruct()
    skin_distance = 0.35
    neighborlist = SimulationCode.initialize_neighbor_struct(skin_distance, box_size, sqrt(force_cutoff2), N, arrays.D_array, dims)
    output.steps_done = timestep
    SimulationCode.update_neighbor_lists!(arrays, parameters, output, neighborlist)
    SimulationCode.calculate_new_forces!(arrays, parameters, neighborlist)
    return arrays, parameters, output, neighborlist
end





using Random, DelimitedFiles

file = "C:\\Users\\s158686\\TU Eindhoven\\Laudicina, Corentin - DATA\\T_0.08000000000000002_seed_626687745_MD.h5"
t = 10^9

arrays, parameters, output, neighborlist = @time read_restart_corentin(file, t);


output.steps_done = 0
parameters.N_steps = 10^6


parameters.dump_info.filename = "Data\\T_0.08000000000000002_seed_626687745_MD_restart.h5"
parameters.dump_info.save=true
parameters.dump_info.when_to_save_array=Set(0:10^3:parameters.N_steps)


@time SimulationCode.perform_molecular_dynamics!(arrays, parameters, output, neighborlist; restarted=false, equilibrate_velocities=false)

