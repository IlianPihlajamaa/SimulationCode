"""
Fills the array with normally distributed numbers for use in the langevin equation
"""
function populate_random_displacement_arrayMD!(arr)
    randn!(arr)
end

function add_new_random_force!(arrays, parameters, system::Langevin)
    γ = system.γ
    Δt = system.Δt
    kBT = system.kBT
    v_array = arrays.v_array
    F_array = arrays.F_array
    rand_num_arr = arrays.random_displacement
    populate_random_displacement_arrayMD!(rand_num_arr)

    factor = sqrt(2 * kBT * γ / Δt)
    for particle_i = 1:parameters.N
        F_array[particle_i] += -γ * v_array[particle_i] + factor * rand_num_arr[particle_i]
    end
end

function add_new_random_force!(arrays, parameters, system::Brownian)
    γ = system.γ
    Δt = system.Δt
    kBT = system.kBT
    F_array = arrays.F_array
    rand_num_arr = arrays.random_displacement
    populate_random_displacement_arrayMD!(rand_num_arr)
    factor = sqrt(2 * kBT * γ / Δt)
    for particle_i = 1:parameters.N
        F_array[particle_i] += factor * rand_num_arr[particle_i]
    end
end

function add_new_random_force!(_, _, S::Newtonian)
    return
end


function compute_forces_berthier_3d(neighborlist, parameters, arrays, particle_i)
    r_array = arrays.r_array
    F_array = arrays.F_array
    D_array = arrays.D_array
    r²_cutoff = parameters.force_cutoff2
    box_size = parameters.box_size
    neighbor_list = neighborlist.neighbor_list_half
    interaction_potential = parameters.interaction_potential
    ri = r_array[particle_i]
    fx = 0.0
    fy = 0.0
    fz = 0.0
    xi = ri[1]
    yi = ri[2]
    zi = ri[3]
    ζ = interaction_potential.ζ
    c2 = interaction_potential.c2
    c4 = interaction_potential.c4
    r_array_r = reinterpret(reshape, Float64, r_array)
    F_array_r = reinterpret(reshape, Float64, F_array)
    Di = D_array[particle_i]
    N_neighbors = neighborlist.neighbor_numbers_half[particle_i]
    @turbo for neighbor_index = 1:N_neighbors
        particle_j = neighbor_list[neighbor_index, particle_i]
        xj = r_array_r[1,particle_j]
        yj = r_array_r[2,particle_j]
        zj = r_array_r[3,particle_j]
        dx = xi - xj
        dy = yi - yj
        dz = zi - zj
        dx -= round(dx / box_size) * box_size
        dy -= round(dy / box_size) * box_size
        dz -= round(dz / box_size) * box_size
        rij2 = dx^2 + dy^2 + dz^2
        Dj = D_array[particle_j]
        mean_d = (Di + Dj)*0.5 * (1.0 - ζ * abs(Di - Dj))  #find_mean_D(Di, Dj, interaction_potential)
        mean_d_squared = mean_d * mean_d
        inv_mean_d2 = 1.0 / mean_d_squared
        invxi2 = rij2*inv_mean_d2#
        invxi4 = invxi2*invxi2#
        xi14 = 1.0/(invxi4*invxi4*invxi4*invxi2)#
        F = ifelse(r²_cutoff * mean_d_squared < rij2, 0.0, -2.0 * inv_mean_d2 * (c2 + 2.0*c4*invxi2 - 6.0*xi14))
        fx += F * dx
        fy += F * dy
        fz += F * dz
        F_array_r[1, particle_j] -= F * dx
        F_array_r[2, particle_j] -= F * dy
        F_array_r[3, particle_j] -= F * dz
    end
    F_array_r[1, particle_i] += fx
    F_array_r[2, particle_i] += fy
    F_array_r[3, particle_i] += fz
end


function compute_forces_berthier_2d(neighborlist, parameters, arrays, particle_i)
    r_array = arrays.r_array
    F_array = arrays.F_array
    D_array = arrays.D_array
    r²_cutoff = parameters.force_cutoff2
    box_size = parameters.box_size
    neighbor_list = neighborlist.neighbor_list_half
    interaction_potential = parameters.interaction_potential
    ri = r_array[particle_i]
    fx = 0.0
    fy = 0.0
    xi = ri[1]
    yi = ri[2]
    ζ = interaction_potential.ζ
    c2 = interaction_potential.c2
    c4 = interaction_potential.c4
    r_array_r = reinterpret(reshape, Float64, r_array)
    F_array_r = reinterpret(reshape, Float64, F_array)
    Di = D_array[particle_i]
    N_neighbors = neighborlist.neighbor_numbers_half[particle_i]
    @turbo for neighbor_index = 1:N_neighbors
        particle_j = neighbor_list[neighbor_index, particle_i]
        xj = r_array_r[1,particle_j]
        yj = r_array_r[2,particle_j]
        dx = xi - xj
        dy = yi - yj
        dx -= round(dx / box_size) * box_size
        dy -= round(dy / box_size) * box_size
        rij2 = dx^2 + dy^2 
        Dj = D_array[particle_j]
        mean_d = (Di + Dj)*0.5 * (1.0 - ζ * abs(Di - Dj))  #find_mean_D(Di, Dj, interaction_potential)
        mean_d_squared = mean_d * mean_d
        inv_mean_d2 = 1.0 / mean_d_squared
        invxi2 = rij2*inv_mean_d2#
        invxi4 = invxi2*invxi2#
        xi14 = 1.0/(invxi4*invxi4*invxi4*invxi2)#
        F = ifelse(r²_cutoff * mean_d_squared < rij2, 0.0, -2.0 * inv_mean_d2 * (c2 + 2.0*c4*invxi2 - 6.0*xi14))
        fx += F * dx
        fy += F * dy
        F_array_r[1, particle_j] -= F * dx
        F_array_r[2, particle_j] -= F * dy
    end
    F_array_r[1, particle_i] += fx
    F_array_r[2, particle_i] += fy
end


"""
Calculates the total force on all particles according to the langevin equation F = -∇U - γv + R. This function also updates the
total potential energy in the output datastructure.
"""
function calculate_new_forces!(arrays, parameters, neighborlist)
    dims = parameters.system.dims

    F_array = arrays.F_array
    Szero = zeros(eltype(F_array))
    for i in eachindex(F_array)
        F_array[i] = Szero
    end 
    r_array = arrays.r_array
    D_array = arrays.D_array
    r²_cutoff = parameters.force_cutoff2
    box_size = parameters.box_size
    neighbor_list = neighborlist.neighbor_list_half
    interaction_potential = parameters.interaction_potential

    for particle_i = 1:parameters.N
        ri = r_array[particle_i]
        Di = D_array[particle_i]
        N_neighbors = neighborlist.neighbor_numbers_half[particle_i]
        if N_neighbors == 0
            continue
        end
        if dims == 3 && typeof(interaction_potential) == Berthier
            compute_forces_berthier_3d(neighborlist, parameters, arrays, particle_i)
        elseif dims == 2 && typeof(interaction_potential) == Berthier
            compute_forces_berthier_2d(neighborlist, parameters, arrays, particle_i)
        else
            f = zero(eltype(r_array))

            for neighbor_index = 1:N_neighbors
                particle_j = neighbor_list[neighbor_index, particle_i]
                rj = r_array[particle_j]
                dr = ri - rj
                dr -= round.(dr / box_size) * box_size
                rij2 = sum(dr.^2)
                Dj = D_array[particle_j]
                mean_d = find_mean_D(Di, Dj, interaction_potential)
                mean_d_squared = mean_d * mean_d
                if r²_cutoff * mean_d_squared > rij2
                    F = force(rij2, mean_d_squared, interaction_potential)
                    f += F * dr
                    F_array[particle_j] -= F * dr
                end
            end
            F_array[particle_i] += f
        end
    end
    # for Brownian and Langevin, add random force (does nothing if Integrator is Newtonian())
    add_new_random_force!(arrays, parameters, parameters.system)

    return
end

"""
Performs an Euler step.
"""
function do_time_step_Euler(arrays, parameters, output, neighborlist, system::Union{Langevin,Newtonian})
    @assert parameters.system.dims == 3

    r_array = arrays.r_array
    v_array = arrays.v_array
    F_array = arrays.F_array
    m = system.m
    displacement_array = neighborlist.displacement_array
    displacement_squared_array = neighborlist.displacement_squared_array

    N = parameters.N
    Δt = system.Δt
    box_size = parameters.box_size

    calculate_new_forces!(arrays, parameters, neighborlist)

    # update velocities 
    for particle_i = 1:N
        v_array[particle_i] = v_array[particle_i] + Δt * F_array[particle_i] / m
    end
    # update positions using periodic boundary conditions
    for particle_i = 1:N
        dr = Δt * v_array[particle_i]
        displacement_array[particle_i] += dr
        displacement_squared_array[particle_i] = sum(displacement_array[particle_i].^2)
        rnew = r_array[particle_i] + dr
        rnew -= floor.(rnew / box_size) * box_size
        r_array[particle_i] = rnew
    end
    if maximum(displacement_squared_array) > (neighborlist.skin_distance / 2.0)^2
        update_neighbor_lists!(arrays, parameters, output, neighborlist)
    end

    # center of mass velocity must be zero.
    vmean = zero(eltype(v_array))
    for particle_i = 1:N
        vmean += v_array[particle_i]
    end
    vmean /= N

    for particle_i = 1:N
        v_array[particle_i] -= vmean
    end

    if system.rescale_temperature # Rescale velocities to set temperature
        rescale_velocities!(v_array, parameters)
    end
end

function rescale_velocities!(v_array, parameters)
    dims = parameters.system.dims

    Ekin = 0.0
    for i = 1:parameters.N
        Ekin += sum(v_array[i] .^ 2)
    end
    Ekin *= parameters.system.m / 2.0
    kBT_current = 2Ekin / (dims * parameters.N)
    factor = sqrt(parameters.system.kBT / kBT_current)
    v_array .*= factor
end


function do_time_step(arrays, parameters, output, neighborlist, system::Brownian)
    @assert parameters.system.dims == 3

    r_array = arrays.r_array
    F_array = arrays.F_array
    displacement_array = neighborlist.displacement_array
    displacement_squared_array = neighborlist.displacement_squared_array

    N = parameters.N
    Δt = system.Δt
    γ = system.γ
    box_size = parameters.box_size

    calculate_new_forces!(arrays, parameters, neighborlist, output)

    # center of mass force must be zero.
    Fmean = zero(eltype(F_array))

    for particle_i = 1:N
        Fmean += F_array[particle_i]
    end
    Fmean /= N

    for particle_i = 1:N
        F_array[particle_i] -= Fmean
    end

    # update positions using periodic boundary conditions
    factor = Δt / γ
    for particle_i = 1:N
        dr = factor * F_array[particle_i]

        displacement_array[particle_i] += dr

        displacement_squared_array[particle_i] = sum(displacement_array[particle_i].^2)
        rnew = r_array[particle_i] + dr
        xnew -= floor.(rnew / box_size) * box_size
        r_array[particle_i] = rnew
    end
    if maximum(displacement_squared_array) > (neighborlist.skin_distance / 2.0)^2
        update_neighbor_lists!(arrays, parameters, output, neighborlist)
    end
end


function do_time_step(arrays, parameters, output, neighborlist, system::Union{Newtonian,Langevin})
    """
    Performs a velocity verlet step.
    """
    r_array = arrays.r_array
    v_array = arrays.v_array
    F_array = arrays.F_array
    m = parameters.system.m
    displacement_array = neighborlist.displacement_array
    displacement_squared_array = neighborlist.displacement_squared_array

    N = parameters.N
    Δt = parameters.system.Δt
    box_size = parameters.box_size
    # update velocities first time
    for particle_i = 1:N
        v_array[particle_i] = v_array[particle_i] + Δt * F_array[particle_i] / (2.0 * m)
    end
    # update positions using periodic boundary conditions
    for particle_i = 1:N
        dr = Δt * v_array[particle_i]
        displacement_array[particle_i] += dr
        displacement_squared_array[particle_i] = sum(displacement_array[particle_i].^2)
        rnew = r_array[particle_i] + dr
        rnew -= floor.(rnew / box_size) * box_size
        r_array[particle_i] = rnew
    end
    if maximum(displacement_squared_array) > (neighborlist.skin_distance / 2.0)^2
        update_neighbor_lists!(arrays, parameters, output, neighborlist)
    end
    calculate_new_forces!(arrays, parameters, neighborlist)

    # update velocities
    for particle_i = 1:N
        v_array[ particle_i] = v_array[particle_i] + Δt * F_array[particle_i] / (2.0 * m)
    end
    # center of mass velocity must be zero.
    vmean = zero(eltype(v_array))

    for particle_i = 1:N
        vmean += v_array[particle_i]
    end
    vmean /= N

    for particle_i = 1:N
        v_array[particle_i] -= vmean
    end
    if system.rescale_temperature
        # Rescale velocities to set temperature
        Ekin = 0.0
        for i = 1:parameters.N
            Ekin += sum(v_array[1, i].^2)
        end
        Ekin *= system.m / 2.0
        kBT_current = 2Ekin / (3 * parameters.N)
        factor = sqrt(system.kBT / kBT_current)
        v_array .*= factor
    end


end

function run_short_MD_equilibration!(arrays, parameters, output, neighborlist)
    """
    Short MD equilibration to relax the relax the velocities and forces
    """
    update_neighbor_lists!(arrays, parameters, output, neighborlist)
    system = parameters.system
    println("Performing short time MD equilibration")
    for i = 1:parameters.N_MD_equilibration_steps
        do_time_step(arrays, parameters, output, neighborlist, system)
    end
end


function print_log_data(arrays, parameters, output, neighborlist)
    output.potential_energy = calculate_full_energy(arrays, parameters, neighborlist)
    if typeof(parameters.system) !== Brownian
        find_kinetic_energy(arrays, parameters, output)
    else
        output.kinetic_energy = 0.0
    end
    if parameters.q_cutoff > 0.0
        calculate_ql_averaged(arrays, parameters, output, neighborlist, 6)
        calculate_ql_averaged(arrays, parameters, output, neighborlist, 4)
    end
    calculate_F2_self!(arrays, parameters, output)
    energy_no_neigh = calculate_full_energy_no_neigh(arrays, parameters)
    if parameters.system.dims == 3
        println(
            "$(output.steps_done)/$(parameters.N_steps), ",
            "E = $(round(output.potential_energy + output.kinetic_energy,digits=2)),  ",
            "E_pot = $(round(output.potential_energy,digits=12)),  ",
            "E_pot_test = $(round(energy_no_neigh, digits=12)),  ",
            "E_kin = $(round(output.kinetic_energy, digits=2)), ",
            "kT = $(round(output.kinetic_energy *2/3/parameters.N, digits=3)), ",
            "q4 = $(round(output.q4,digits=3)), ",
            "q6 = $(round(output.q6, digits=3)), ",
            "F₂s = $(round(output.F2s, digits=3))"
        )
    else 
        println(
            "$(output.steps_done)/$(parameters.N_steps), ",
            "E = $(round(output.potential_energy + output.kinetic_energy,digits=2)),  ",
            "E_pot = $(round(output.potential_energy,digits=12)),  ",
            "E_pot_test = $(round(energy_no_neigh, digits=12)),  ",
            "E_kin = $(round(output.kinetic_energy, digits=2)), ",
            "kT = $(round(output.kinetic_energy *2/3/parameters.N, digits=3)), ",
            "F₂s = $(round(output.F2s, digits=3))"
        )
    end

end


function perform_molecular_dynamics!(arrays, parameters, output, neighborlist; restarted=false, equilibrate_velocities=true)
    """
    Main MD loop. 
    """
    println("\n\nStarting MD procedure")
    arrays.r_old_array .= arrays.r_array
    if !restarted && equilibrate_velocities && typeof(parameters.system) !== Brownian 
        randn!(arrays.v_array)
        arrays.v_array .*= sqrt(parameters.system.kBT / parameters.system.m)
    end
    if !restarted
        output.steps_done = 0
    end
    println("Running production MD")
    update_neighbor_lists!(arrays, parameters, output, neighborlist)
    if equilibrate_velocities
        run_short_MD_equilibration!(arrays, parameters, output, neighborlist)
        rescale_velocities!(arrays.v_array, parameters)
    end
    output.potential_energy = calculate_full_energy(arrays, parameters, neighborlist)
    output.N_neighbor_list_rebuilds = 0

    prepare_savefile(parameters, arrays)
    save_data(arrays, parameters, output, neighborlist, restarted)
    parameters.callback(arrays, parameters, output, neighborlist)

    @time while output.steps_done < parameters.N_steps
        do_time_step(arrays, parameters, output, neighborlist, parameters.system)
        output.steps_done += 1
        parameters.callback(arrays, parameters, output, neighborlist)
        save_data(arrays, parameters, output, neighborlist, restarted)
        if logdata(output.steps_done)
            print_log_data(arrays, parameters, output, neighborlist)
            if output.q6 > 0.2
                error("Simulation has crystallized, terminating...")
            end
        end
    end
    println("MD procedure complete")
    total_Neighbor_builds = output.N_neighbor_list_rebuilds
    println("Number of neighbor lists built = $total_Neighbor_builds")

end
