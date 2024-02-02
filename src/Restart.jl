function read_restart(file, timestep)
    f = h5open(file) 
    attribs = attrs(f)
    r_array = read(f, "positions/$(timestep)")
    v_array = read(f, "velocities/$(timestep)")
    F_array = read(f, "forces/$(timestep)")
    D_array = read(f, "diameters/diameters")

    Δt = attribs["Δt"]
    filename = attribs["filename"]
    ρ = attribs["ρ"]
    N = attribs["N"]
    box_size = attribs["box_size"]
    swap_probability = attribs["swap_probability"]
    max_MC_displacement = attribs["max_MC_displacement"]
    force_cutoff2 = attribs["force_cutoff2"]
    N_steps = attribs["N_steps"]
    q_cutoff = attribs["q_cutoff"]
    interaction_potential = attribs["interaction_potential"]
    c0 = attribs["c0"]
    c2 = attribs["c2"]
    c4 = attribs["c4"]
    dims = attribs["dims"]
    random_seed = attribs["random_seed"]

    save = attribs["save"]
    save_r = attribs["save_r"]
    save_v = attribs["save_v"]
    save_F = attribs["save_F"]
    save_D = attribs["save_D"]
    save_Epot = attribs["save_Epot"]
    when_to_save_array = attribs["when_to_save_array"]



    kBT = attribs["kBT"]
    m = attribs["m"]
    rescale_temperature = attribs["rescale_temperature"]
    N_MD_equilibration_steps = attribs["N_MD_equilibration_steps"]
    ζ = attribs["ζ"]
    σ_ratio = attribs["σ_ratio"]

    system = attribs["system"]


    close(f)
    arrays = ArrayStruct{dims}(
        Vector(reinterpret(reshape, SVector{dims, Float64}, r_array)), 
        Vector(reinterpret(reshape, SVector{dims, Float64}, copy(r_array))),
        D_array,
        Vector(reinterpret(reshape, SVector{dims, Float64}, v_array)), 
        Vector(reinterpret(reshape, SVector{dims, Float64}, F_array)), 
        [zeros(SVector{dims}) for _ in 1:N],
        zeros(N)
        )

    if split(interaction_potential, '.')[end] == "Berthier"
        interaction_potential = Berthier(
        c0,
        c2, 
        c4, 
        ζ, 
        σ_ratio)
    end
    if split(system, '.')[end] == "Newtonian"
        system = Newtonian(
        kBT,
        Δt,
        m,
        rescale_temperature,
        dims)
    end
    dump_info = SimulationCode.DumpInfo(
        save, #save
        filename,
        Set(when_to_save_array), #when save
        save_r, #r
        save_v, #v
        save_F, #F
        save_D, #D
        save_Epot, #Epot      
    )

    cb(x...) = nothing 
    parameters = ParameterStruct(
        N_MD_equilibration_steps,
        random_seed,
        ρ,
        N,
        box_size,
        N_steps,
        swap_probability,
        max_MC_displacement,
        force_cutoff2,
        q_cutoff,
        system,
        interaction_potential,
        dump_info,
        cb
    )
    output = OutputStruct()
    skin_distance = 0.35
    neighborlist = SimulationCode.initialize_neighbor_struct(skin_distance, box_size, sqrt(force_cutoff2), N, arrays.D_array, dims)
    output.steps_done = timestep

    return arrays, parameters, output, neighborlist
end
