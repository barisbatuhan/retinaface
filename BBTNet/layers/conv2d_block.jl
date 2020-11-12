include("core.jl")

"""
Defines a block of Conv2D layers stacked into each other.
    layers      : An array of Conv2D layers stacked sequentially
    input_dim   : An integer that gets the channel dimension of the input image
    output_dim  : An integer that keeps the filter size of the last Conv2D layer
"""
mutable struct Conv2D_Block
    layers::Array{Any};
    input_dim::Int64;
    output_dim::Int64;
end

"""
Custom constructor for Conv2D_Block.
    input_dim       : An integer that gets the channel dimension of the input image
    layer_count     : Number of layers stacked on each other
    kernel_sizes    : Array of tuples, each tuple holds x and y dimensions of a kernel.
                    If the size of this array and number of layers are not equal to each other,
                    then for each block, the element corresponding (layer_count % array_size + 1)
                    is taken as the kernel size.
    filters         : Filter size for each single layer. Modulo operation explained in kernel_sizes
                    case also applies for this parameter.
    paddings        : Padding size for each single layer. Modulo operation explained in kernel_sizes
                    case also applies for this parameter.
    strides         : Stride size for each single layer. Modulo operation explained in kernel_sizes
                    case also applies for this parameter.
    dilations       : Dilation size for each single layer. Modulo operation explained in kernel_sizes
                    case also applies for this parameter.
    activations     : Activation function for each single layer. Modulo operation explained in kernel_sizes
                    case also applies for this parameter.
    dropout         : An integer value that sets the dropout ratio for each of the layers in the network.
"""
function Conv2D_Block(
    input_dim::Int64,
    layer_count::Int64, 
    kernel_sizes::Array{Tuple{Int64, Int64}},
    filters::Array{Int64};
    paddings::Array{Int64}=[0],
    strides::Array{Int64}=[1],
    dilations::Array{Int64}=[1],
    activations::Array{String}=["relu"],
    dropout=0, atype=Array{Float64}
    )

    layers = []
    output_dim = -1

    for iter in 1:layer_count 
        kernel_size = kernel_sizes[mod(iter - 1, length(kernel_sizes)) + 1]
        activation = _get_activation(activations[mod(iter - 1, length(activations)) + 1])
        filter_size = filters[mod(iter - 1, length(filters)) + 1]
        pad = paddings[mod(iter - 1, length(paddings)) + 1]
        stride = strides[mod(iter - 1, length(strides)) + 1]
        dilation = dilations[mod(iter - 1, length(dilations)) + 1]
        if length(layers) == 0
            push!(layers, Conv2D(
                kernel_size[1], kernel_size[2], input_dim, filter_size, 
                f=activation, pdrop=dropout, padding=pad,
                stride=stride, dilation=dilation))
        else
            last_filter = filters[mod(iter - 2, length(filters)) + 1]
            push!(layers, Conv2D(
                kernel_size[1], kernel_size[2], last_filter, 
                filter_size, f=activation, pdrop=dropout, padding=pad,
                stride=stride, dilation=dilation, atype=atype))
        end
        if iter == layer_count output_dim = filter_size end
    end
    return Conv2D_Block(layers, input_dim, output_dim)
end

# Conv2D Block Forward Pass
function (block::Conv2D_Block)(x)
    val = block.layers[1](x)
    for layer in block.layers[2:end]
        val = layer(val)
    end
    return val
end

function _get_activation(str::String)
    str = lowercase(str)
    if str == "relu"
        return relu
    elseif str == "sigmoid"
        return sigm
    elseif str == "elu"
        return elu
    else
        return nothing
    end
end