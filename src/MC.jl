"""
Performs a translation step for a given particle in a Monte Carlo simulation.

# Arguments
- `arrays`: Struct containing arrays of particle positions, displacements, etc.
- `parameters`: Struct containing simulation parameters.
- `output`: Struct for tracking simulation output metrics.
- `neighborlist`: Struct for neighbor list data.
- `particle_i`: Index of the particle to be moved.

# Details
This function attempts to move a particle by a random displacement, calculates the change in potential energy, and decides whether to accept the move based on the Boltzmann factor. If the move is accepted, the particle's position is updated and relevant metrics are adjusted.

The function also updates the displacement arrays and counts the number of accepted and total translation attempts.

# Example
```julia
do_translation_step!(arrays, parameters, output, neighborlist, particle_i)
```
"""
function do_translation_step!(arrays, parameters, output, neighborlist, particle_i)
    box_size = parameters.box_size
    r_array = arrays.r_array
    D_array = arrays.D_array
    kBT = parameters.system.kBT
    displacement_array = neighborlist.displacement_array
    displacement_squared_array = neighborlist.displacement_squared_array
    rold = r_array[particle_i]
    Dold = D_array[particle_i]
    E_old = find_energy(particle_i, rold, Dold, arrays, parameters, neighborlist)
    dr = arrays.random_displacement[particle_i]
    rnew = rold + dr
    rnew -= floor.(rnew/box_size)*box_size
    Dnew = Dold
    E_new = find_energy(particle_i, rnew, Dnew, arrays, parameters, neighborlist)
    dE = E_new - E_old
    boltzman_factor = min(1.0, exp(-dE/kBT))
    output.N_translations_total += 1
    myrand = rand()
    if boltzman_factor > myrand
        output.N_translations_accepted += 1
        r_array[particle_i] = rnew
        displacement_array[particle_i] += dr
        displacement_squared_array[particle_i] = sum(displacement_array[particle_i].^2)
        output.potential_energy += dE
    end
end

"""
    do_swap_step!(arrays, parameters, output, neighborlist, particle_i, particle_j)

Perform a Monte Carlo swap step between two particles in a simulation.

This function attempts to swap the diameters of two particles within a given simulation box,
calculating the change in potential energy and accepting or rejecting the swap based on the
Metropolis criterion.

# Arguments
- `arrays::Arrays`: An object containing arrays storing information about particle positions and diameters.
- `parameters::Parameters`: An object containing various simulation parameters, such as system temperature and box size.
- `output::Output`: An object tracking the total number of swaps attempted and accepted, and updating the potential energy.
- `neighborlist::NeighborList`: A list of neighboring particles for efficient energy calculation.
- `particle_i::Int`: The index of the first particle involved in the swap.
- `particle_j::Int`: The index of the second particle involved in the swap.

# Details
- If `particle_i` is the same as `particle_j`, the function returns immediately without making any changes.
- The function only proceeds with the swap if the absolute difference between the diameters of the two particles is less than or equal to 0.2.
- Computes the energy of the system before and after the swap using the `find_energy` function.
- The swap is accepted based on the Boltzmann factor calculated from the energy change, otherwise it is rejected.
- Updates the `output` object with the number of swaps attempted, number of swaps accepted, and the change in potential energy if the swap is accepted.

# Returns
- `Nothing`: The function modifies the input arrays and output object in place.
"""
function do_swap_step!(arrays, parameters, output, neighborlist, particle_i, particle_j)

    if particle_j == particle_i
        return
    end
    box_size = parameters.box_size
    r_array = arrays.r_array
    D_array = arrays.D_array
    kBT = parameters.system.kBT
    riold = r_array[ particle_i]
    Diold = D_array[particle_i]
    rjold = r_array[particle_j]
    Djold = D_array[particle_j]

    if abs(Diold - Djold) > 0.2
        return
    end
    Ei_old = find_energy(particle_i, riold, Diold, arrays, parameters, neighborlist)
    Ej_old = find_energy(particle_j, rjold, Djold, arrays, parameters, neighborlist)
    Dinew, Djnew = Djold, Diold
    D_array[particle_i] = Dinew
    D_array[particle_j] = Djnew
    Ei_new = find_energy(particle_i, riold, Dinew, arrays, parameters, neighborlist)
    Ej_new = find_energy(particle_j, rjold, Djnew, arrays, parameters, neighborlist)
    dE = (Ei_new + Ej_new) - (Ei_old + Ej_old)
    boltzman_factor = min(1.0, exp(-dE/kBT))
    output.N_swaps_total += 1
    if boltzman_factor > rand()
        output.N_swaps_accepted += 1
        output.potential_energy += dE
    else
        D_array[particle_i] = Diold
        D_array[particle_j] = Djold
    end
    return
end

"""
    do_MC_step!(arrays, parameters, output, neighborlist, swap_probability)

Perform a Monte Carlo step in the simulation.

This function performs a Monte Carlo step in the simulation, which can include translation or swap steps
for each particle in the system. The type of step is determined probabilistically based on the given
`swap_probability`.

# Arguments
- `arrays::Arrays`: An object containing arrays storing information about particle positions and diameters.
- `parameters::Parameters`: An object containing various simulation parameters, such as system temperature and box size.
- `output::Output`: An object tracking the total number of swaps attempted and accepted, and updating the potential energy.
- `neighborlist::NeighborList`: A list of neighboring particles for efficient energy calculation.
- `swap_probability::Float64`: The probability of performing a swap step instead of a translation step.

# Details
- Generates a random displacement array for each particle using `populate_random_displacement_array!` function.
- For each particle, decides whether to perform a translation or a swap step based on `swap_probability`.
- If a translation step is chosen, calls the `do_translation_step!` function for that particle.
- If the displacement of the particle exceeds half the skin distance, updates the neighbor list.
- If a swap step is chosen, randomly selects another particle to swap with and calls the `do_swap_step!` function.

# Returns
- `Nothing`: The function modifies the input arrays and output object in place.
"""
function do_MC_step!(arrays, parameters, output, neighborlist, swap_probability)
    displacement_squared_array = neighborlist.displacement_squared_array
    N = parameters.N


    populate_random_displacement_array!(arrays.random_displacement, N, parameters.max_MC_displacement)
    for particle_i = 1:N
        if rand() > swap_probability # do a translation step
            do_translation_step!(arrays, parameters, output, neighborlist, particle_i)
            if displacement_squared_array[particle_i] > (neighborlist.skin_distance/2.0)^2
                update_neighbor_lists!(arrays, parameters, output, neighborlist)
            end
        else #do swap step
            particle_j = ceil(Int64, rand()/N)
            do_swap_step!(arrays, parameters, output, neighborlist, particle_i, particle_j)
        end
    end
end

"""
    perform_swap_monte_carlo!(arrays, parameters, output, neighborlist)

Perform a Monte Carlo simulation with swap moves.

This function executes a Monte Carlo simulation, incorporating swap moves between particles.
It iterates over a specified number of steps and performs various actions such as updating neighbor lists,
calculating energies, and saving data. Additionally, it checks for crystallization during the simulation.

# Arguments
- `arrays::Arrays`: An object containing arrays storing information about particle positions and diameters.
- `parameters::Parameters`: An object containing various simulation parameters, such as system temperature and box size.
- `output::Output`: An object tracking the simulation progress and results.
- `neighborlist::NeighborList`: A list of neighboring particles for efficient energy calculation.

# Details
- Initializes the simulation and sets up necessary variables and counters.
- Executes the Monte Carlo steps using the `do_MC_step!` function.
- Calls user-defined callback functions and saves data periodically.
- Checks for crystallization during the simulation and terminates if detected.
- Prints information about the simulation progress and results.

# Returns
- `Nothing`: The function prints simulation information to the console.
"""
function perform_swap_monte_carlo!(arrays, parameters, output, neighborlist; stepsdone=0)
    dump_info = parameters.dump_info
    update_neighbor_lists!(arrays, parameters, output, neighborlist)
    output.kinetic_energy = calculate_full_energy(arrays, parameters, neighborlist)
    output.steps_done = stepsdone
    output.N_swaps_total = 0
    output.N_swaps_accepted = 0
    output.N_translations_total = 0
    output.N_translations_accepted = 0
    output.N_neighbor_list_rebuilds = 0
    
    arrays.r_old_array .= arrays.r_array

    if stepsdone == 0 # otherwise we are restarting file already exists
        prepare_savefile(parameters, arrays)
        save_data(arrays, parameters, output, neighborlist, false)
    end
    
    parameters.callback(arrays, parameters, output, neighborlist)
    output.potential_energy = calculate_full_energy_no_neigh(arrays, parameters)
    start_t = time()
    @time while output.steps_done <= parameters.N_steps
        do_MC_step!(arrays, parameters, output, neighborlist, parameters.swap_probability)
        parameters.callback(arrays, parameters, output, neighborlist)
        output.steps_done += 1
        save_data(arrays, parameters, output, neighborlist, false)
        if logdata(output.steps_done)
            if parameters.q_cutoff > 0.0
                calculate_ql_averaged(arrays, parameters, output, neighborlist, 6)
                calculate_ql_averaged(arrays, parameters, output, neighborlist, 4)
            end
            calculate_F2_self!(arrays, parameters, output)
            energy_no_neigh = calculate_full_energy_no_neigh(arrays, parameters)
            t_still = estimated_remaining_time(output.steps_done, start_t, parameters.N_steps)
            println(
                        "$(output.steps_done)/$(parameters.N_steps), ",
                        "ETA = $(round(t_still, digits=1)), ", 
                        "E_pot = $(round(output.potential_energy,digits=12)),  ",
                        "E_pot_test = $(round(energy_no_neigh, digits=12)),  ",
                        "q4 = $(round(output.q4,digits=3)), ", 
                        "q6 = $(round(output.q6, digits=3)), ",
                        "F₂s = $(round(output.F2s, digits=3))"
                    )
            flush(stdout)

            if output.q6 > 0.2
                error("Simulation has crystallized, terminating...") 
            end    
        end


    end
    println("Completed the MC procedure.")
    N_neighbor_list_rebuilds = output.N_neighbor_list_rebuilds
    total_swaps = output.N_swaps_total
    accepted_swaps = output.N_swaps_accepted
    total_translations = output.N_translations_total
    accepted_translations = output.N_translations_accepted
    println("Number of neighbor lists built = $N_neighbor_list_rebuilds")
    println("Swap moves accepted: $accepted_swaps/$total_swaps")
    println("Translation moves accepted: $accepted_translations/$total_translations")
end

