import Pkg; Pkg.activate(".")

include("src/SimulationCode.jl")

using Random, DelimitedFiles
T = "0.08000000000000002"
seed = 626687745
file = "Data\\T_$(T)_seed_$(seed)_MD_restart.h5"
t1 = 1000
t2 = 3000

arrays1, parameters1, output1, neighborlist1 = @time SimulationCode.read_restart(file, t1);

norm(x) = sqrt(sum(x.^2))
maxnorm(x::Vector{T}) where T = maximum(norm.(x))
maxnorm(x) = maximum(norm(x[:, i]) for i in axes(x,2))

original_when_to_save_set = parameters1.dump_info.when_to_save_array

function create_callback(particle_list::Vector{Int}, original_when_to_save_set)
    function cb2(arrays, parameters, output, neighborlist)
        if output.steps_done in original_when_to_save_set
            SimulationCode.update_neighbor_lists!(arrays, parameters, output, neighborlist)
        end
         if output.steps_done in parameters.dump_info.when_to_save_array
            SimulationCode.compute_potential_energy!(arrays, parameters, output, neighborlist, particle_list)
            for particle in particle_list
                filename = parameters.dump_info.filename[1:end-3]*"_p_$(particle).txt"
                open(filename, "a") do f
                    print(f, output.steps_done, " ", arrays.r_array[particle][1], " ", arrays.r_array[particle][2], " ", arrays.r_array[particle][3], " ")
                    print(f, arrays.v_array[particle][1], " ", arrays.v_array[particle][2], " ", arrays.v_array[particle][3], " ")
                    print(f, arrays.F_array[particle][1], " ", arrays.F_array[particle][2], " ", arrays.F_array[particle][3], " ")
                    print(f, arrays.potential_energy_array[particle], "\n")
                end
            end
        end
    end
    return cb2
end




## recreate with new callback
parameters1 = SimulationCode.ParameterStruct(
    parameters1.N_MD_equilibration_steps, 
    parameters1.random_seed, 
    parameters1.ρ, 
    parameters1.N, 
    parameters1.box_size, 
    parameters1.N_steps, 
    parameters1.swap_probability, 
    parameters1.max_MC_displacement, 
    parameters1.force_cutoff2, 
    parameters1.q_cutoff, 
    parameters1.system, 
    parameters1.interaction_potential, 
    parameters1.dump_info, 
    create_callback(Int[1,2,3,4], original_when_to_save_set))

parameters1.N_steps = t2


parameters1.dump_info.filename = file[1:end-3]*"_restart_t1_$t1.h5"
parameters1.dump_info.save = false
parameters1.dump_info.when_to_save_array = Set(t1:t2)


SimulationCode.perform_molecular_dynamics!(arrays1, parameters1, output1, neighborlist1; restarted=true, equilibrate_velocities=false)



arrays2, parameters2, output2, neighborlist2 = @time SimulationCode.read_restart(file, t2);


println("compare t1-t2 restart")
@assert maxnorm(arrays1.r_array .- arrays2.r_array) == 0.0
@assert maxnorm(arrays1.v_array .- arrays2.v_array) == 0.0
@assert maxnorm(arrays1.F_array .- arrays2.F_array) == 0.0


t_array =  0:1000:10^6
r_array = zeros(3, 1200, length(t_array))
r_array_min = zeros(3, 1200, length(t_array))
using Base.Threads
for it = eachindex(t_array)
    t = t_array[it]
    @show it
    arrays, parameters, output, neighborlist = @time SimulationCode.read_restart(file, t);
    r_array[:, :, it] .= reinterpret(reshape, Float64, copy(arrays.r_array))


    r_array_to_be_minimized = copy(arrays.r_array)

    @time  r_minimized = find_inherent_structure(r_array_to_be_minimized, parameters1, arrays1, neighborlist1, output1)

    r_array_min[:, :, it] .= reinterpret(reshape, Float64, r_minimized)
end

using Plots
for i = 1:1000
for dim = 1:3
    if 1.0 < maximum(r_array[dim, i, :]) - minimum(r_array[dim, i, :]) < 5.0
    p = plot() 
    plot!(t_array, r_array[dim, i, :], color=:blue, label="real r")
    plot!(t_array, r_array_min[dim, i, :], color=dim, label="inherent r")|> display
    end
end
end

@show "end"
using JLD2
save_object("Data/r_array_test_T_$T.jld2", r_array)
save_object("Data/r_array_inherent_test_T_$T.jld2", r_array_min)