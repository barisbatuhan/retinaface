using Knet

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
function Dense(i::Int, o::Int; f=nothing, pdrop=0, dtype=Array{Float64}, bias=true) 
    b = nothing
    if bias b = Param(convert(dtype, zeros(o, 1))) end
    return Dense(Param(convert(dtype, randn(o, i))), b, f, pdrop)
end

# Dense Layer Forward Pass
function (d::Dense)(x; train=true)
    if train
        x_val = d.w * mat(dropout(x,d.p))
        if d.b !== nothing x_val .+= d.b end
        if d.f !== nothing x_val = d.f.(x_val) end
        return x_val
    else
        x_val = value(d.w) * mat(x)
        if d.b !== nothing x_val .+= value(d.b) end
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
    w1::Int, w2::Int, input_dim::Int, output_dim::Int; f=nothing, 
    pdrop=0, padding=0, stride=1, dilation=1, bias=true, dtype=Array{Float64}
    ) 
    b = nothing
    if bias b = Param(convert(dtype, zeros(1, 1, output_dim, 1))) end
    
    return Conv2D(
        Param(convert(dtype, xavier(w1, w2, input_dim, output_dim))),
        b, f, pdrop, padding, stride, dilation
        )
end

# Conv2D Layer Forward Pass
function (c::Conv2D)(x; train=true)
    # apply dropout only in training
    if train
        x = dropout(x, c.p)
        x = conv4(c.w, x, padding=c.padding, stride=c.stride, dilation=c.dilation) 
        if c.b !== nothing x .+= c.b end
    else
        x = conv4(value(c.w), x, padding=c.padding, stride=c.stride, dilation=c.dilation) 
        if c.b !== nothing x .+= value(c.b) end
    end
    if c.f === nothing return x
    else return c.f.(x)
    end
end

"""
Batch Normalization Layer
- works both for convolutional and fully connected layers
"""
mutable struct BatchNorm bn_params; end

function BatchNorm(;momentum=0.1) 
    return BatchNorm(bnmoments(momentum=momentum))
end

function (bn::BatchNorm)(x; train=true)
    return batchnorm(x, bn.bn_params; training=train)
end



