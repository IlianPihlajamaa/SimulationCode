
function update_neighbor_lists!(arrays, parameters, output, neighborlist)
    """
    Constructs verlet lists by a cell decomposition of the simulation box. Much faster than the full loop if N>>10^2.
    """
    
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


