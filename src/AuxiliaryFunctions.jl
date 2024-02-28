
function calculate_full_energy(arrays, parameters, neighborlist)
    """Calculates the potential energy of the system"""
    energy = 0.0
    r_array = arrays.r_array
    D_array = arrays.D_array
    N = parameters.N
    box_size = parameters.box_size
    @inbounds for particle_i = 1:N
        ri = r_array[particle_i]
        Di = D_array[particle_i]
        energy += find_energy(particle_i, ri, Di, arrays, parameters, neighborlist)
    end
    return energy / 2.0 #double counting
end

function find_kinetic_energy(arrays, parameters, output)
    """Calculates the kinetic energy of the system"""
    v_array = arrays.v_array
    Ekin = 0.0
    for i = 1:parameters.N
        Ekin += sum(v_array[i].^2)
    end
    Ekin *= parameters.system.m / 2.0
    output.kinetic_energy = Ekin
end


function calculate_full_energy_no_neigh(arrays, parameters)


    """Calculates the potential energy of a single particle"""

    interaction_potential = parameters.interaction_potential
    box_size = parameters.box_size
    r²_cutoff = parameters.force_cutoff2
    energy = 0.0
    N = parameters.N
    r_array = arrays.r_array
    D_array = arrays.D_array
    for particle_i = 1:N
        ri = r_array[particle_i]
        Di = D_array[particle_i]
        for particle_j = 1:N
            if particle_i == particle_j
                continue
            end
            rj = r_array[particle_j]
            Dj = D_array[particle_j]
            rij2 = calculate_squared_distance(ri,rj, box_size)
            mean_d = find_mean_D(Di, Dj, interaction_potential)
            mean_d_squared = mean_d * mean_d
            energy += ifelse(r²_cutoff * mean_d_squared < rij2, 0.0, potential(rij2, mean_d_squared, interaction_potential))
        end
    end
    return energy / 2.0
end


function calculate_energy_no_neigh_i(particle_i, ri, Di, arrays, parameters)
    """Calculates the potential energy of a single particle"""
    interaction_potential = parameters.interaction_potential
    box_size = parameters.box_size
    r²_cutoff = parameters.force_cutoff2
    energy = 0.0
    N = parameters.N
    r_array = arrays.r_array
    D_array = arrays.D_array
    for particle_j = 1:N
        if particle_i == particle_j
            continue
        end
        rj = r_array[particle_j]
        Dj = D_array[particle_j]
        rij2 = calculate_squared_distance(ri, rj, box_size)
        mean_d = find_mean_D(Di, Dj, interaction_potential)
        mean_d_squared = mean_d * mean_d
        energy += ifelse(r²_cutoff * mean_d_squared < rij2, 0.0, potential(rij2, mean_d_squared, interaction_potential))
    end

    return energy 
end

function compute_potential_energy!(arrays, parameters, output, neighborlist)
    r_array = arrays.r_array
    D_array = arrays.D_array
    potential_energy_array = arrays.potential_energy_array
    for particle_i in 1:parameters.N
        ri = r_array[particle_i]
        Di = D_array[particle_i]
        Ei = find_energy(particle_i, ri, Di, arrays, parameters, neighborlist)
        potential_energy_array[particle_i] = Ei/2.0
    end
end

function compute_potential_energy!(arrays, parameters, output, neighborlist, particle_list)
    r_array = arrays.r_array
    D_array = arrays.D_array
    potential_energy_array = arrays.potential_energy_array
    for particle_i in particle_list
        ri = r_array[particle_i]
        Di = D_array[particle_i]
        Ei = find_energy(particle_i, ri, Di, arrays, parameters, neighborlist)
        potential_energy_array[particle_i] = Ei/2.0
    end
end

function find_volume_fraction(Di, box_size, dims)
    # @assert dims == 3
    Vbox = box_size^dims
    if dims == 1
        Vparticles = 0.0
        for D in Di
            Vparticles += D
        end
        return Vparticles / Vbox
    elseif dims == 2
        Vparticles = 0.0
        for D in Di
            Vparticles += π * D ^ 2 / 4
        end
        return Vparticles / Vbox
    elseif dims == 3
        Vparticles = 0.0
        for D in Di
            Vparticles += π * D ^ 3 / 6
        end
        return Vparticles / Vbox
    elseif dims==4
        Vparticles = 0.0
        for D in Di
            Vparticles += π^2 * D ^ 4 / 32
        end
        return Vparticles / Vbox
    end
    error("This is dimension is not implemented")
end

@inline function calculate_squared_distance(ri, rj, box_size)

    """
    Calculates squared distance between particles taking the periodic boundary conditions into account
    """
    dr = ri - rj
    dr -= round.(dr / box_size) * box_size
    return sum(dr.^2)
end

function GC_inner_loop(interaction_potential, r_array, ri, N_neighbors, particle_i, D_array, box_size, r²_cutoff, neighbor_list)
    ϵ = interaction_potential.ϵ
    Uc = interaction_potential.Uc
    xi = ri[1]
    yi = ri[2]
    zi = ri[3]
    r_array_r = reinterpret(reshape, Float64, r_array)
    energy = 0.0
    @turbo for neighbor_index = 1:N_neighbors
        particle_j = neighbor_list[neighbor_index, particle_i]
        xj = r_array_r[1, particle_j]
        yj = r_array_r[2, particle_j]
        zj = r_array_r[3, particle_j]
        dx = xi - xj
        dy = yi - yj
        dz = zi - zj
        dx -= round(dx / box_size) * box_size
        dy -= round(dy / box_size) * box_size
        dz -= round(dz / box_size) * box_size
        rij2 = dx^2 + dy^2 + dz^2
        invxi2 = rij2
        energy += ifelse(r²_cutoff < rij2, 0.0, ϵ*exp(-invxi2) - Uc)
    end
    return energy

end

function berthier_inner_loop3(interaction_potential, r_array, ri, N_neighbors, particle_i, D_array, box_size, r²_cutoff, neighbor_list)
    ζ = interaction_potential.ζ
    c0 = interaction_potential.c0
    c2 = interaction_potential.c2
    c4 = interaction_potential.c4
    xi = ri[1]
    yi = ri[2]
    zi = ri[3]
    Di = D_array[particle_i]
    r_array_r = reinterpret(reshape, Float64, r_array)
    energy = 0.0
    for neighbor_index = 1:N_neighbors
        particle_j = neighbor_list[neighbor_index, particle_i]
        xj = r_array_r[1, particle_j]
        yj = r_array_r[2, particle_j]
        zj = r_array_r[3, particle_j]
        Dj = D_array[particle_j]
        dx = xi - xj
        dy = yi - yj
        dz = zi - zj
        dx -= round(dx / box_size) * box_size
        dy -= round(dy / box_size) * box_size
        dz -= round(dz / box_size) * box_size
        rij2 = dx^2 + dy^2 + dz^2
        mean_d = (Di + Dj)*0.5 * (1.0 - ζ * abs(Di - Dj)) 
        mean_d_squared = mean_d * mean_d
        invxi2 = rij2/mean_d_squared#
        invxi4 = invxi2*invxi2#
        xi12 = 1.0/(invxi4*invxi4*invxi4)#
        energy += ifelse(r²_cutoff * mean_d_squared < rij2, 0.0, xi12 + c4 * invxi4 + c2*invxi2 + c0)
    end
    return energy
end


function berthier_inner_loop2(interaction_potential, r_array, ri, N_neighbors, particle_i, D_array, box_size, r²_cutoff, neighbor_list)
    ζ = interaction_potential.ζ
    c0 = interaction_potential.c0
    c2 = interaction_potential.c2
    c4 = interaction_potential.c4
    xi = ri[1]
    yi = ri[2]
    Di = D_array[particle_i]
    r_array_r = reinterpret(reshape, Float64, r_array)
    energy = 0.0
    @turbo for neighbor_index = 1:N_neighbors
        particle_j = neighbor_list[neighbor_index, particle_i]
        xj = r_array_r[1, particle_j]
        yj = r_array_r[2, particle_j]
        Dj = D_array[particle_j]
        dx = xi - xj
        dy = yi - yj
        dx -= round(dx / box_size) * box_size
        dy -= round(dy / box_size) * box_size
        rij2 = dx^2 + dy^2 
        mean_d = (Di + Dj)*0.5 * (1.0 - ζ * abs(Di - Dj)) 
        mean_d_squared = mean_d * mean_d
        invxi2 = rij2/mean_d_squared#
        invxi4 = invxi2*invxi2#
        xi12 = 1.0/(invxi4*invxi4*invxi4)#
        energy += ifelse(r²_cutoff * mean_d_squared < rij2, 0.0, xi12 + c4 * invxi4 + c2*invxi2 + c0)
    end
    return energy
end

function IPL_inner_loop(interaction_potential, r_array, ri, N_neighbors, particle_i, D_array, box_size, r²_cutoff, neighbor_list)
    ϵ = interaction_potential.ϵ
    σ = interaction_potential.σ
    n = interaction_potential.n
    Uc = interaction_potential.Uc
    xi = ri[1]
    yi = ri[2]
    zi = ri[3]
    r_array_r = reinterpret(reshape, Float64, r_array)
    energy = 0.0
    @turbo for neighbor_index = 1:N_neighbors
        particle_j = neighbor_list[neighbor_index, particle_i]
        xj = r_array_r[1, particle_j]
        yj = r_array_r[2, particle_j]
        zj = r_array_r[3, particle_j]
        dx = xi - xj
        dy = yi - yj
        dz = zi - zj
        dx -= round(dx / box_size) * box_size
        dy -= round(dy / box_size) * box_size
        dz -= round(dz / box_size) * box_size
        rij2 = dx^2 + dy^2 + dz^2
        mean_d = σ 
        mean_d_squared = mean_d * mean_d
        xi2 = mean_d_squared/rij2#
        xin = xi2^(n/2)
        energy += ifelse(r²_cutoff * mean_d_squared < rij2, 0.0, ϵ*xin-Uc)
    end
    return energy
end

function LJ_inner_loop(interaction_potential, r_array, ri, N_neighbors, particle_i, D_array, box_size, r²_cutoff, neighbor_list)
    ϵ = interaction_potential.ϵ
    σ = interaction_potential.σ
    Uc = interaction_potential.Uc
    xi = ri[1]
    yi = ri[2]
    zi = ri[3]
    r_array_r = reinterpret(reshape, Float64, r_array)
    energy = 0.0
    @turbo for neighbor_index = 1:N_neighbors
        particle_j = neighbor_list[neighbor_index, particle_i]
        xj = r_array_r[1, particle_j]
        yj = r_array_r[2, particle_j]
        zj = r_array_r[3, particle_j]
        Dj = D_array[particle_j]
        dx = xi - xj
        dy = yi - yj
        dz = zi - zj
        dx -= round(dx / box_size) * box_size
        dy -= round(dy / box_size) * box_size
        dz -= round(dz / box_size) * box_size
        rij2 = dx^2 + dy^2 + dz^2
        mean_d = σ 
        mean_d_squared = mean_d * mean_d
        xi2 = mean_d_squared/rij2#
        xi6 = xi2*xi2*xi2
        xi12 = xi6*xi6
        energy += ifelse(r²_cutoff * mean_d_squared < rij2, 0.0, 4ϵ*(xi12-xi6)-Uc)
    end
    return energy
end

function find_energy(particle_i, ri, Di, arrays, parameters, neighborlist)
    """Calculates the potential energy of a single particle"""
    box_size = parameters.box_size
    interaction_potential = parameters.interaction_potential
    r²_cutoff = parameters.force_cutoff2
    energy = 0.0
    N_neighbors = neighborlist.neighbor_numbers_full[particle_i]
    if N_neighbors == 0
        return 0.0
    end

    r_array = arrays.r_array
    D_array = arrays.D_array
    neighbor_list = neighborlist.neighbor_list_full
    if parameters.system.dims == 3 && typeof(interaction_potential)==Berthier
        return berthier_inner_loop3(interaction_potential, r_array, ri, N_neighbors, particle_i, D_array, box_size, r²_cutoff, neighbor_list)
    elseif parameters.system.dims == 2 && typeof(interaction_potential)==Berthier
        return berthier_inner_loop2(interaction_potential, r_array, ri, N_neighbors, particle_i, D_array, box_size, r²_cutoff, neighbor_list)
    elseif parameters.system.dims == 3 && typeof(interaction_potential)==GaussianCore
        return GC_inner_loop(interaction_potential, r_array, ri, N_neighbors, particle_i, D_array, box_size, r²_cutoff, neighbor_list)
    elseif parameters.system.dims == 3 && typeof(interaction_potential)==LJ
        return LJ_inner_loop(interaction_potential, r_array, ri, N_neighbors, particle_i, D_array, box_size, r²_cutoff, neighbor_list)
    elseif parameters.system.dims == 3 && typeof(interaction_potential)==PowerLaw
        return IPL_inner_loop(interaction_potential, r_array, ri, N_neighbors, particle_i, D_array, box_size, r²_cutoff, neighbor_list)
    else
        @inbounds for neighbor_index = 1:N_neighbors
            particle_j = neighbor_list[neighbor_index, particle_i]
            rj = r_array[particle_j]
            Dj = D_array[particle_j]
            rij2 = calculate_squared_distance(ri, rj, box_size)
            mean_d = find_mean_D(Di, Dj, interaction_potential)
            mean_d_squared = mean_d * mean_d
            if r²_cutoff * mean_d_squared > rij2
                energy += potential(rij2, mean_d_squared, interaction_potential)
            end
        end
    end
    return energy
end

"""
Finds potential energy of one pair of particles
"""
function find_pair_energy(ri,rj, Dj, box_size, parameters)
    r²_cutoff = parameters.force_cutoff2
    interaction_potential = parameters.interaction_potential
    rij2 = calculate_squared_distance(ri,rj, box_size)
    mean_d = find_mean_D(Di, Dj, interaction_potential)
    mean_d_squared = mean_d * mean_d
    energy += ifelse(r²_cutoff * mean_d_squared < rij2, 0.0, potential(rij2, mean_d_squared, interaction_potential))
    return energy
end



@inline function populate_random_displacement_array!(a, N, max_displacement)
    """
    Populates displacement array for translation monte calo step
    """
    rand!(a)
    for j = 1:N
        a[j] = a[j] * 2 * max_displacement .- max_displacement
    end
end

function logdata(t::Int)
    if t == 0
        return true
    end
    log10t = log10(t)
    if log10t >= 2 && t % 10^floor(Int64, log10t) == 0
        return true
    end
    return false
end


function calculate_ql(arrays, parameters, output, neighborlist, l)
    @assert parameters.system.dims == 3

    """
    Calculates the non-averaged Steinhardt order parameters
    """
    #q6_total(i) = sqrt(4pi/(2l+1)  sum_m |q6m(i)|^2)
    #q6m(i) = 1/Nneigh sum_j=neigh Y6m(rij)
    q6_total_averaged = 0.0
    for particle_i = 1:parameters.N
        q6_total = 0.0
        for m = -l:l
            q6_total += abs(calculate_qlm(arrays, parameters, neighborlist, particle_i, l, m))^2
        end
        q6_total = sqrt(4π / (2l + 1) * q6_total)
        q6_total_averaged += q6_total
    end
    q6_total_averaged /= parameters.N
    if l == 6
        output.q6 = q6_total_averaged
    elseif l == 4
        output.q4 = q6_total_averaged
    else
        println(q6_total_averaged)
    end
end


function calculate_qlm(arrays, parameters, neighborlist, particle_i, l, m)
    @assert parameters.system.dims == 3

    """
    Calculates qlm = 1/Nneigh sum_j=neigh Y6m(rij) for particle_i
    """
    r_array = arrays.r_array
    box_size = parameters.box_size
    xi = r_array[particle_i][1]
    yi = r_array[particle_i][2]
    zi = r_array[particle_i][3]
    N_neighbors = neighborlist.neighbor_numbers_full[particle_i]
    neighbor_list = neighborlist.neighbor_list_full
    q_cutoff2 = parameters.q_cutoff^2
    q6m = 0.0 + 0.0im
    Y6m = 0.0 + 0.0im
    N_neighborsq = 0
    @inbounds for neighbor_index = 1:N_neighbors
        particle_j = neighbor_list[neighbor_index, particle_i]
        xj = r_array[particle_j][1]
        yj = r_array[particle_j][2]
        zj = r_array[particle_j][3]
        xij = xi - xj
        yij = yi - yj
        zij = zi - zj
        xij -= round(xij / box_size) * box_size
        yij -= round(yij / box_size) * box_size
        zij -= round(zij / box_size) * box_size
        if xij^2 + yij^2 + zij^2 > q_cutoff2
            continue
        end
        N_neighborsq += 1
        @fastmath theta = atan(sqrt(xij^2 + yij^2) / zij)
        if xij == 0.0
            xij += 10^-10
        end

        @fastmath phi = atan(yij / xij)

        Y6m = SphericalHarmonics.sphericalharmonic(theta, phi, l=l, m=m)
        q6m += Y6m
    end
    if N_neighborsq == 0
        return N_neighborsq, q6m
    else
        q6m /= N_neighborsq
        return N_neighborsq, q6m
    end
end



function calculate_ql_averaged(arrays, parameters, output, neighborlist, l)
    @assert parameters.system.dims == 3

    """
    Calculates the averaged Steinhardt order parameters
    """
    qlm_array = zeros(Complex{Float64}, 2l + 1, parameters.N)
    N_neighbors_array = zeros(Int64, 2l + 1, parameters.N)
    r_array = arrays.r_array
    box_size = parameters.box_size
    q_cutoff2 = parameters.q_cutoff^2
    q6_total_averaged = 0.0
    for particle_i = 1:parameters.N
        for m = -l:l
            N_Neighbors, q6_totalk = calculate_qlm(arrays, parameters, neighborlist, particle_i, l, m)
            qlm_array[m+l+1, particle_i] = q6_totalk
            N_neighbors_array[m+l+1, particle_i] = N_Neighbors
        end
    end

    for particle_i = 1:parameters.N
        xi = r_array[particle_i][1]
        yi = r_array[particle_i][2]
        zi = r_array[particle_i][3]
        q6_total = 0.0
        for m = -l:l
            N_neighbors = neighborlist.neighbor_numbers_full[particle_i]
            neighbor_list = neighborlist.neighbor_list_full
            q6_totalk = qlm_array[m+l+1, particle_i]
            for neighbor_index = 1:N_neighbors
                particle_j = neighbor_list[neighbor_index, particle_i]
                xj = r_array[particle_j][1]
                yj = r_array[particle_j][2]
                zj = r_array[particle_j][3]
                xij = xi - xj
                yij = yi - yj
                zij = zi - zj
                xij -= round(xij / box_size) * box_size
                yij -= round(yij / box_size) * box_size
                zij -= round(zij / box_size) * box_size
                if xij^2 + yij^2 + zij^2 > q_cutoff2
                    continue
                end
                q6_totalk += qlm_array[m+l+1, particle_j]
            end
            N_neighborsq = N_neighbors_array[m+l+1, particle_i]
            q6_totalk /= (N_neighborsq + 1)
            q6_total += abs(q6_totalk)^2
        end
        q6_total = sqrt(q6_total * 4π / (2l + 1))
        q6_total_averaged += q6_total
    end

    q6_total_averaged /= parameters.N
    if l == 6
        output.q6 = q6_total_averaged
    elseif l == 4
        output.q4 = q6_total_averaged
    else
        println(q6_total_averaged)
    end
end


function find_exponential_time_array(stop, factor)
    """
    returns an array of exponentially spaced integers
    """
    t_arr = Int64[]
    push!(t_arr, 0)
    t = 1
    while t < stop
        push!(t_arr, t)
        t = ceil(Int64, t * factor)
    end
    return t_arr
end

function calculate_F2_self!(arrays, parameters, output)

    rold = arrays.r_old_array
    r = arrays.r_array
    kx = 7.2

    N = length(rold)
    F2s = 0.0
    for i = 1:N
        rx = rold[i][1]
        krold = kx * rx 
        sinkrold, coskrold = sincos(krold)
        rx = r[i][1]
        kr = kx * rx 
        sinkr, coskr = sincos(kr)
        F2s += coskrold * coskr + sinkrold * sinkr
    end
    F2s /= N
    output.F2s = F2s
end
