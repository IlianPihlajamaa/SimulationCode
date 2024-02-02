
"""
finds inherent structure by naive minimization. will mutate the arrays input.
"""
function find_inherent_structure(r_array_to_be_minimized, parameters, arrays, neighborlist, output; tolerance=10^-6, max_iterations=10^6, max_step_size=0.001, verbose=1)
    r_array = arrays.r_array
    arrays.r_array .= r_array_to_be_minimized
    error = tolerance*2
    SimulationCode.update_neighbor_lists!(arrays, parameters, output, neighborlist)
    dt = 0.005#parameters.system.Δt^2
    box_size = parameters.box_size
    displacement_array = neighborlist.displacement_array
    displacement_squared_array = neighborlist.displacement_squared_array
    displacement_array .*= 0.0
    displacement_squared_array .*= 0.0
    iteration = 0
    F_array = arrays.F_array
    v_array = arrays.v_array

    while error > tolerance
        iteration += 1
        if iteration > max_iterations 
            error("Not converged")
        end

        SimulationCode.calculate_new_forces!(arrays, parameters, neighborlist)
        error = 0.0

        for particle_i in 1:parameters.N
            v_array[particle_i] = F_array[particle_i]*dt
        end

        stepsize = 0.0
        for particle_i in 1:parameters.N
            stepsize = max(stepsize, sqrt(sum(v_array[particle_i].^2)))
        end

        if stepsize > max_step_size
            v_array .*= max_step_size/stepsize
        end
        for particle_i in 1:parameters.N
            dr = v_array[particle_i]
            displacement_array[particle_i] += dr
            displacement_squared_array[particle_i] = sum(displacement_array[particle_i].^2)
            rnew = r_array[particle_i] + dr
            rnew -= floor.(rnew / box_size) * box_size
            r_array[particle_i] = rnew
        end
        if maximum(displacement_squared_array) > (neighborlist.skin_distance / 2.0)^2
            SimulationCode.update_neighbor_lists!(arrays, parameters, output, neighborlist)
        end
        maxforce = 0.0
        for particle_i in 1:parameters.N
            maxforce = max(maxforce, sqrt(sum(F_array[particle_i].^2)))
        end
        error = max(maxforce, stepsize)
        if verbose >= 2 && iteration % 1000 == 1
            SimulationCode.compute_potential_energy!(arrays, parameters, output, neighborlist)

            maxchange = 0.0
            for i = 1:parameters.N
                ri = r_array_to_be_minimized[i]
                rj = r_array[i]
                box_size = parameters.box_size
                rij2 = SimulationCode.calculate_squared_distance(ri, rj, box_size)
                maxchange = max(maxchange, sqrt(rij2))
            end
            @show iteration, maxchange, stepsize, maxforce, sum(arrays.potential_energy_array)

        end
    end
    SimulationCode.compute_potential_energy!(arrays, parameters, output, neighborlist)

    maxchange = 0.0
    if verbose >= 1
        for i = 1:parameters.N
            ri = r_array_to_be_minimized[i]
            rj = r_array[i]
            box_size = parameters.box_size
            rij2 = SimulationCode.calculate_squared_distance(ri, rj, box_size)
            maxchange = max(maxchange, sqrt(rij2))
        end
        println("after ", iteration, 
        " iterations, the potential energy is ", sum(arrays.potential_energy_array),
         ". Maximal total displacement = ", maxchange)
    end
    return copy(r_array)
end