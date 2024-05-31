

# Import packages and files

module SimulationCode

using DelimitedFiles, Random, HDF5, SphericalHarmonics
using IfElse, LoopVectorization, CellListMap, StaticArrays

for file in [
    "DataStructs.jl", 
    "AuxiliaryFunctions.jl",
    "InitialConfiguration.jl", 
    "InteractionPotential.jl", 
    "NeighborLists.jl", 
    "MC.jl", 
    "MD.jl", 
    "Restart.jl", 
    "Dump.jl", 
    "InherentStructure.jl"
    ] 
    
    include(file) 
end


end
