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
function Dense(i::Int, o::Int; init=xavier_uniform, f=nothing, pdrop=0, dtype=Array{Float32}, bias=true) 
    b = nothing
    if bias b = Param(convert(dtype, zeros(o, 1))) end
    return Dense(Param(convert(dtype, init(o, i))), b, f, pdrop)
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
    w1::Int, w2::Int, input_dim::Int, output_dim::Int; init=xavier_uniform, f=nothing, 
    pdrop=0, padding=0, stride=1, dilation=1, bias=true, dtype=Array{Float32}
    ) 
    b = nothing
    if bias b = Param(convert(dtype, zeros(1, 1, output_dim, 1))) end
    
    return Conv2D(
        Param(convert(dtype, init(w1, w2, input_dim, output_dim))),
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
mutable struct BatchNorm bn_moments; bn_params; end

function BatchNorm(;channels=1, momentum=0.1, dtype=Array{Float32}) 
    return BatchNorm(
        bnmoments(momentum=momentum),
        convert(dtype, bnparams(Float32, channels))
    )
end

function (bn::BatchNorm)(x; train=true)  
    return batchnorm(x, bn.bn_moments, bn.bn_params; training=train)
end


# struct BatchNormLayer; w; ms; 

#     function BatchNormLayer(pre_w, pre_ms)
#         res_mean = popfirst!(pre_ms)
#         # Trick to arrange variance value for new(er) batchnorm
#         res_variance =   popfirst!(pre_ms).^2  .- 1e-5
#         ms = bnmoments(mean=res_mean, var=res_variance)
    
#         w1 = pre_w[1]
#         w2 = pre_w[2]
#         w1 = vec(w1)
#         w2 = vec(w2)
#         w =  vcat(w1, w2)
#         param_w = param(w, atype=Knet.atype())
#         return new(param_w, ms)
#     end

# end

# function (batch_norm_layer::BatchNormLayer)(x)
#     return batchnorm(x, batch_norm_layer.ms, batch_norm_layer.w; eps=1e-5)
# end









