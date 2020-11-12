include("../layers/conv2d_block.jl")

struct ResNet50
    conv_block_1
    conv_block_2
    conv_block_3
    conv_block_4
    conv_block_5
end

# function ResNet50(;input_dim::Array{Int64}, include_top::Bool=true)
        
# end