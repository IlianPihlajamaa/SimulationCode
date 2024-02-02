function do_translation_step!(arrays, parameters, output, neighborlist, particle_i)

    """
    performs a translation step of particle_i. calculates the current energy of that particle and the energy after a random displacement. 
    Accepts the step with the metropolis rule.
    """
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

function do_swap_step!(arrays, parameters, output, neighborlist, particle_i, particle_j)

    """
    performs a swap step of particle_i and particle_j. calculates the combined current energy of the particles and their energy after a random displacement. 
    Accepts the step with the metropolis rule.
    """
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


function do_MC_step!(arrays, parameters, output, neighborlist, swap_probability)
    """
    Performs a single MC step for all particles. Decides randomly which type of step should be performed.
    """
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


function perform_swap_monte_carlo!(arrays, parameters, output, neighborlist)
    """
    Main MC loop.
    """
    println("\n\nStarting MC procedure")
    dump_info = parameters.dump_info
    update_neighbor_lists!(arrays, parameters, output, neighborlist)
    output.kinetic_energy = calculate_full_energy(arrays, parameters, neighborlist)
    output.steps_done = 0
    output.N_swaps_total = 0
    output.N_swaps_accepted = 0
    output.N_translations_total = 0
    output.N_translations_accepted = 0
    output.N_neighbor_list_rebuilds = 0
    # when_to_save_array = create_when_to_save_array(parameters.N_steps, 50)
    prepare_savefile(parameters, arrays)
    arrays.r_old_array .= arrays.r_array
    save_data(arrays, parameters, output, neighborlist, false)
    parameters.callback(arrays, parameters, output, neighborlist)
    output.potential_energy = calculate_full_energy_no_neigh(arrays, parameters)

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
            println(
                        "$(output.steps_done)/$(parameters.N_steps), ",
                        "E_pot = $(round(output.potential_energy,digits=12)),  ",
                        "E_pot_test = $(round(energy_no_neigh, digits=12)),  ",
                        "q4 = $(round(output.q4,digits=3)), ", 
                        "q6 = $(round(output.q6, digits=3)), ",
                        "F₂s = $(round(output.F2s, digits=3))"
                    )

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

