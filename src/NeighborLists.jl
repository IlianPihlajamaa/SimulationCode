"""
    update_neighbor_lists!(arrays, parameters, output, neighborlist)

Update neighbor lists using cell decomposition.

This function updates neighbor lists using a cell decomposition technique, which is faster than a full pairwise loop for large systems.
It constructs Verlet lists by decomposing the simulation box into cells and updating neighbor lists based on particle positions.

# Arguments
- `arrays::Arrays`: An object containing arrays storing information about particle positions.
- `parameters::Parameters`: An object containing various simulation parameters, such as force cutoff distance.
- `output::Output`: An object tracking the simulation progress and results.
- `neighborlist::NeighborList`: A list of neighboring particles for efficient force calculation.

# Details
- Increments the neighbor list rebuild counter in the output object.
- Updates the cell list using particle positions and auxiliary structures.
- Resets displacement arrays and neighbor numbers for full and half neighbor lists.
- Constructs neighbor lists using pairwise interactions within the force cutoff distance.

# Returns
- `Nothing`: The function modifies the neighbor list arrays in place.
"""
function update_neighbor_lists!(arrays, parameters, output, neighborlist)
    output.N_neighbor_list_rebuilds += 1

    r_array = arrays.r_array
    box = neighborlist.box
    celllist = neighborlist.celllist
    auxilliary_struct = neighborlist.auxilliary_struct    
    celllist = UpdateCellList!(r_array, box, celllist, auxilliary_struct, parallel=false)
    
    interaction_potential = parameters.interaction_potential
    # neighborlist.neighbor_list_full .= 0
    neighborlist.neighbor_numbers_full .= 0
    # neighborlist.neighbor_list_half .= 0
    neighborlist.neighbor_numbers_half .= 0
    Szero = zeros(eltype(neighborlist.displacement_array))
    for i in eachindex(neighborlist.displacement_array)
        neighborlist.displacement_array[i] = Szero
    end 
    neighborlist.displacement_squared_array .= 0.0

    r_cutoff = sqrt(parameters.force_cutoff2)

    map_pairwise!(
        (x,y,i,j,d2,pairs) -> add_pair!(i,j,d2,neighborlist, arrays, r_cutoff, interaction_potential),
        neighborlist,
        box,
        celllist,
        parallel=false
    )
end


"""
    add_pair!(i, j, d2, neighborlist, arrays, r_cutoff, interaction_potential)

Add a pair of particles to the neighbor lists.

This function adds a pair of particles to the full and half neighbor lists based on the distance between them and the force cutoff distance.
It calculates the mean diameter between the particles based on their individual diameters and the interaction potential.

# Arguments
- `i::Int`: Index of the first particle.
- `j::Int`: Index of the second particle.
- `d2::Float64`: Squared distance between the particles.
- `neighborlist::NeighborList`: A list of neighboring particles for efficient force calculation.
- `arrays::Arrays`: An object containing arrays storing information about particle diameters.
- `r_cutoff::Float64`: Force cutoff distance.
- `interaction_potential`: The interaction potential between particles.

# Details
- Increments the neighbor numbers for both full and half neighbor lists for both particles.
- Adds the index of the second particle to the neighbor list of the first particle if it falls within the cutoff distance.
- Calculates the mean diameter between the particles and updates the half neighbor list accordingly.
- Updates the neighbor list of the second particle in a similar manner.

# Returns
- `Nothing`: The function modifies the neighbor list arrays in place.
"""
function add_pair!(i,j,d2, neighborlist, arrays, r_cutoff, interaction_potential)
    neighbor_number = neighborlist.neighbor_numbers_full[i]
    neighbor_number += 1
    neighborlist.neighbor_numbers_full[i] = neighbor_number
    neighborlist.neighbor_list_full[neighbor_number, i] = j

    Di = arrays.D_array[i]
    Dj = arrays.D_array[j]
    mean_d = find_mean_D(Di, Dj, interaction_potential)

    if d2 < ((r_cutoff * mean_d) + neighborlist.skin_distance)^2 # for MD the diameters do not change, so we can include particles within a smaller range
        neighbor_number = neighborlist.neighbor_numbers_half[i]
        neighbor_number += 1
        neighborlist.neighbor_numbers_half[i] = neighbor_number
        neighborlist.neighbor_list_half[neighbor_number, i] = j
    end

    neighbor_number = neighborlist.neighbor_numbers_full[j]
    neighbor_number += 1
    neighborlist.neighbor_numbers_full[j] = neighbor_number
    neighborlist.neighbor_list_full[neighbor_number, j] = i
    return
end


