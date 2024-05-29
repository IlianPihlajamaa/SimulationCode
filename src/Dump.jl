"""
    create_when_to_save_array(maxsteps, doublefactor)

Creates an array of steps at which data should be saved during a simulation.

This function generates an array of integers representing the steps at which 
data should be saved. The intervals between these steps increase exponentially.

# Arguments
- `maxsteps`: Maximum number of simulation steps.
- `doublefactor`: Factor by which the interval (`dt`) between saves increases.

# Returns
- `save_array`: Array of integers indicating the steps at which to save data.

# Details
- The function starts with a step interval (`dt`) of 1 and doubles the interval 
  after `doublefactor` steps.
- This array helps in saving data more frequently at the beginning and less 
  frequently as the simulation progresses, which is useful for capturing early 
  dynamics in detail while reducing the amount of saved data for later steps.

"""
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

"""
    save_data(arrays, parameters::ParameterStruct{A,B}, output, neighborlist, restarted) where {A,B}

Saves the relevant information to disk.

This function saves various arrays and computed data to an HDF5 file at specified 
intervals, ensuring the ability to restart the simulation and analyze the data.

# Arguments
- `arrays`: The ArrayStruct containing particle information and other arrays.
- `parameters`: The ParameterStruct containing simulation parameters and settings.
- `output`: The OutputStruct containing computed outputs from the simulation.
- `neighborlist`: The NeighborListStruct for managing neighbor lists.
- `restarted`: Boolean flag indicating if the simulation was restarted.

# Details
- Saves data only if the current step is in the `when_to_save_array` of `dump_info`.
- If not restarted, updates the neighbor lists to ensure reproducibility.
- Saves particle positions, velocities, forces, diameters, and potential energy 
  as per the settings in `dump_info`.

# Usage
The function checks the settings in `dump_info` to determine which arrays to save.
The data is saved in a hierarchical HDF5 file with paths based on the number of 
steps completed.
"""
function save_data(arrays, parameters::ParameterStruct{A,B}, output, neighborlist, restarted) where {A,B}
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



"""
    prepare_savefile(parameters, arrays)

Prepares a savefile by storing attributes to the HDF5 file.

This function initializes an HDF5 file and creates the necessary groups for storing
simulation data. It also saves the simulation parameters as attributes within the file.

# Arguments
- `parameters`: The ParameterStruct containing simulation parameters and settings.
- `arrays`: The ArrayStruct containing particle information and other arrays.

# Details
- The function checks if saving is enabled in `dump_info`. If not, it exits early.
- Creates groups in the HDF5 file for positions, velocities, forces, diameters,
  potential energy, and miscellaneous data.
- Stores attributes from `ParameterStruct`, the system, interaction potential,
  and `dump_info` (excluding `when_to_save_array`) in the HDF5 file.
- Saves the `D_array` (diameters) under the `diameters` group.

# Usage
This function ensures that all relevant parameters and initial data are saved 
in the HDF5 file, providing a comprehensive record of the simulation setup.
"""
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