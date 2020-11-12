using Knet

"""
A classic fully connected layer.
    w   : Weight matrix
    b   : Bias vector
    f   : activation function
    p   : dropout ratio
"""
mutable struct Dense
    w 
    b 
    f 
    p
end

"""
A custom constructor for Dense
    i       : Dimension of the input
    o       : Dimension of the input
    f       : Activation function
    pdrop   : Dropout ratio
"""
function Dense(i::Int, o::Int, f=relu; pdrop=0, atype=Array{Float64}) 
    return Dense(
        Param(convert(atype, randn(o, i))),
        Param(convert(atype, zeros(o, 1))), f, pdrop)
end

# Dense Layer Forward Pass
function (d::Dense)(x)
    return d.w * mat(dropout(x,d.p)) .+ d.b
end


"""
A convolutional layer with 2D kernel dimensions.
    w   : Weight matrix
    b   : Bias vector
    f   : activation function
    p   : dropout ratio
"""
mutable struct Conv2D 
    w 
    b 
    f 
    p 
    padding 
    stride 
    dilation 
end

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
    w1::Int, w2::Int, input_dim::Int, output_dim::Int; f=relu, 
    pdrop=0, padding=0, stride=1, dilation=1, atype=Array{Float64}
    ) 
    return Conv2D(
        Param(convert(atype, randn(w1, w2, input_dim, output_dim))),
        Param(convert(atype, zeros(1, 1, output_dim, 1))),
        f, pdrop, padding, stride, dilation
        )
end

# Conv2D Layer Forward Pass
function (c::Conv2D)(x)
    return c.f.(conv4(
        c.w, dropout(x,c.p), 
        padding=c.padding, 
        stride=c.stride, 
        dilation=c.dilation) .+ c.b)
end