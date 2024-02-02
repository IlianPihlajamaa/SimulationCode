
function create_when_to_save_array(maxsteps, doublefactor)
    save_array = Int64[]
    t = 0
    dt = 1
    while t <= maxsteps
        push!(save_array, t)
        t += dt
        if t == dt * doublefactor
            dt *= 10
        end
    end
    sort!(save_array)
    return save_array
end

function save_data(arrays, parameters::ParameterStruct{A,B}, output, neighborlist, restarted) where {A,B}
    """
    Saves the relevant information to disk
    """
    dump_info = parameters.dump_info
    stepsdone = output.steps_done

    if !(dump_info.save && stepsdone in dump_info.when_to_save_array)
        return
    elseif !restarted 
        update_neighbor_lists!(arrays, parameters, output, neighborlist) # for complete reproducibility
    end

    file = h5open(dump_info.filename, "cw") # IOstream variable
    if dump_info.save_r 
        file["positions/"*string(stepsdone)] = reinterpret(reshape, Float64, arrays.r_array)
    end
    if dump_info.save_v 
        file["velocities/"*string(stepsdone)] = reinterpret(reshape, Float64, arrays.v_array)
    end
    if dump_info.save_F 
        file["forces/"*string(stepsdone)] = reinterpret(reshape, Float64, arrays.F_array)
    end
    if dump_info.save_D 
        file["diameters/"*string(stepsdone)] = arrays.D_array
    end
    if dump_info.save_Epot 
        compute_potential_energy!(arrays, parameters, output, neighborlist)
        file["potential_energy/"*string(stepsdone)] = arrays.potential_energy_array
    end
    close(file)
end




function prepare_savefile(parameters, arrays)
    """
    Prepares a savefile by storing attributes to the h5 file
    """
    if !parameters.dump_info.save 
        return
    end

    file = h5open(parameters.dump_info.filename, "w") # IOstream variable

    create_group(file, "positions")
    create_group(file, "velocities")
    create_group(file, "forces")
    create_group(file, "diameters")
    create_group(file, "potential_energy")
    create_group(file, "miscellaneous")

    for fieldname in fieldnames(ParameterStruct)
        if !(fieldname in (:file, :system, :interaction_potential, :dump_info, :callback)) 
            attributes(file)[string(fieldname)] = getfield(parameters, fieldname)
        else
            attributes(file)[string(fieldname)] = string(typeof(getfield(parameters, fieldname)))
        end
    end
    for fieldname in fieldnames(typeof(parameters.system))
        attributes(file)[string(fieldname)] = getfield(parameters.system, fieldname)
    end
    for fieldname in fieldnames(typeof(parameters.interaction_potential))
        attributes(file)[string(fieldname)] = getfield(parameters.interaction_potential, fieldname)
    end
    for fieldname in fieldnames(typeof(parameters.dump_info))
        if fieldname == :when_to_save_array 
            continue
        else
            attributes(file)[string(fieldname)] = getfield(parameters.dump_info, fieldname)
        end
    end
    file["diameters/"*"diameters"] = arrays.D_array


    close(file)
end