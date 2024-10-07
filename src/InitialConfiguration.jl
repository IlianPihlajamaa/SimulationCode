function generate_diameters!(arrays, parameters, U::Berthier)
    # Sample regularly from the P(D) = A/D^3 distribution using inverse of CDF
    D_array = arrays.D_array
    N = length(D_array)
    σ_ratio = U.σ_ratio
    A = 1/2 + 1/(σ_ratio-1)
    σmin = (1 + σ_ratio)/(2*σ_ratio)
    D_array .= shuffle([sqrt(1/(σmin^-2 - 2P/A)) for P in LinRange(1/(2N), 1-1/(2N), N)])

end

function generate_diameters!(arrays, parameters, U::Union{LJ, PowerLaw, GaussianCore})
    # all diameters are equal
    D_array = arrays.D_array
    D_array .= ones(parameters.N)*U.σ
end

function generate_diameters!(arrays, parameters, U::HardSphere)
    # all diameters are equal
    D_array = arrays.D_array
    D_array .= 1.0
    println("The volume fraction is φ = $(find_volume_fraction(D_array, parameters.box_size, parameters.system.dims))")
end


function generate_diameters!(arrays, parameters, U::Weysser)
    # Sample regularly from the uniform distribution P(D) = 1/(2δ),  D ∈ (1-δ, 1+δ)
    D_array = arrays.D_array
    N = length(D_array)
    δ = U.δ
    D_array .= shuffle([2*δ*P+1-δ for P in LinRange(1/(2N), 1-1/(2N), N)])
    println("The volume fraction is φ = $(find_volume_fraction(D_array, parameters.box_size, parameters.system.dims))")
end

function sample_A_sigma3(σmin, A)
    P = rand()
    σ = sqrt(1/(σmin^-2 - 2P/A))
    return σ
end

function find_random_initial_configuration!(arrays, parameters, output, neighborlist; max_growth_steps=10000, steps=1000)
    """
    Places N particles on a lattice and does some MC steps to make it energetically physical
    """
    dims = parameters.system.dims
    N = parameters.N
    kBT = parameters.system.kBT
    parameters.system.kBT = 10.0
    box_size = parameters.box_size
    r_array = arrays.r_array
    println("The volume fraction is φ = $(find_volume_fraction(arrays.D_array, parameters.box_size, parameters.system.dims))")
    println("The temperature is kBT = $(parameters.system.kBT)")

    # place on cubic lattice
    Nx = ceil(Int64, N^(1/dims))
    for i = 1:N
        r_array[i] = rand(SVector{dims,Float64})*box_size 
    end
    output.steps_done = 0
    update_neighbor_lists!(arrays, parameters, output, neighborlist)
    output.potential_energy = calculate_full_energy(arrays, parameters, neighborlist)
    println("Initialized particles on a simple cubic lattice.")
    println("Running quick MC equilibration at high temperature.")
    if !isfinite(output.potential_energy) && parameters.interaction_potential == HardSphere()
        
        println("Growing the hard spheres when overlaps")
        arrays.D_array .*= 0.01
        for i=0:max_growth_steps
            ## Increase diameters
            update_neighbor_lists!(arrays, parameters, output, neighborlist)
            output.potential_energy = calculate_full_energy(arrays, parameters, neighborlist)

            if arrays.D_array[1] == 1.0 && isfinite(output.potential_energy)
                break
            end

            if isfinite(output.potential_energy)
                arrays.D_array .+= 0.01            
                if arrays.D_array[1] > 1.0
                    arrays.D_array .= 1.0
                end
            end

            do_MC_step!(arrays, parameters, output, neighborlist, 0.0)
        end

        if arrays.D_array[1] < 1.0 || !isfinite(output.potential_energy)
            error("Not coverged, increase growth steps")
        end
    end

    @time for i=0:steps
        if i % (steps ÷ 10) == 0
            output.potential_energy = calculate_full_energy(arrays, parameters, neighborlist)
            if parameters.q_cutoff > 0.0
                calculate_ql_averaged(arrays, parameters, output, neighborlist, 6)
                calculate_ql_averaged(arrays, parameters, output, neighborlist, 4)
            end
            println(
                        "$(i)/$(steps), ",
                        "E_pot = $(round(output.potential_energy,digits=2)),  ",
                        "q4 = $(round(output.q4,digits=6)), ", 
                        "q6 = $(round(output.q6, digits=6))"
                    )
                         
        end
        do_MC_step!(arrays, parameters, output, neighborlist, 0.0)
    end
    println("Successfully found an initial configuration. Starting simulation now.")
    parameters.system.kBT = kBT
end

