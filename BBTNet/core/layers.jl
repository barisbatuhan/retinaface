using Knet

include("initializers.jl")

"""
A classic fully connected layer.
    w   : Weight matrix
    b   : Bias vector
    f   : activation function
    p   : dropout ratio
"""
mutable struct Dense w; b; f; p; end

"""
A custom constructor for Dense
    i       : Dimension of the input
    o       : Dimension of the input
    f       : Activation function
    pdrop   : Dropout ratio
"""
function Dense(i::Int, o::Int; init=xavier, f=nothing, pdrop=0, dtype=Array{Float32}, bias=true) 
    b = convert(dtype, zeros(o, 1))
    if bias b = Param(b) end
    return Dense(Param(convert(dtype, init(o, i))), b, f, pdrop)
end

# Dense Layer Forward Pass
function (d::Dense)(x; train=true)
    if train
        x_val = d.w * mat(dropout(x,d.p)) .+ d.b
        if d.f !== nothing x_val = d.f.(x_val) end
        return x_val
    else
        x_val = value(d.w) * mat(x) .+ value(d.b)
        if d.f !== nothing x_val = d.f.(x_val) end
        return x_val
    end 
end

"""
A convolutional layer with 2D kernel dimensions.
    w   : Weight matrix
    b   : Bias vector
    f   : activation function
    p   : dropout ratio
"""
mutable struct Conv2D w; b; f; p; padding; stride; dilation; end

"""
A custom constructor for Conv2D
    w1          : First dimension of the kernel
    w2          : Second dimension of the kernel
    input_dim   : Channel dimension of the input
    output_dim  : Filter size
    f           : Activation function
    pdrop       : Dropout ratio
    padding     : Padding size
    stride      : Stride size
    dilation    : Dilation size
"""
function Conv2D(
    w1::Int, w2::Int, input_dim::Int, output_dim::Int; init=kaiming_uniform, f=nothing, 
    pdrop=0, padding=0, stride=1, dilation=1, bias=true, dtype=Array{Float32}
    ) 
    b = convert(dtype, zeros(1, 1, output_dim, 1))
    if bias b = Param(b) end
    
    return Conv2D(
        Param(convert(dtype, init(w1, w2, input_dim, output_dim))),
        b, f, pdrop, padding, stride, dilation
        )
end

# Conv2D Layer Forward Pass
function (c::Conv2D)(x; train=true)
    # apply dropout only in training
    if train
        if c.p > 0 x = dropout(x, c.p) end
        x = conv4(c.w, x, padding=c.padding, stride=c.stride, dilation=c.dilation) .+ c.b
    else
        x = conv4(value(c.w), x, padding=c.padding, stride=c.stride, dilation=c.dilation) .+ value(c.b)
    end
    if c.f === nothing return x
    else return c.f.(x)
    end
end

"""
Batch Normalization Layer
- works both for convolutional and fully connected layers
"""
mutable struct BatchNorm bn_moments; bn_params; end

function BatchNorm(;channels=1, momentum=0.1, dtype=Array{Float32}) 
    return BatchNorm(
        bnmoments(momentum=momentum),
        Param(convert(dtype, bnparams(Float32, channels)))
    )
end

function (bn::BatchNorm)(x; train=true)  
    return batchnorm(x, bn.bn_moments, bn.bn_params; training=train, eps=1e-5)
end










