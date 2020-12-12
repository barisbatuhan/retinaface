include("layers.jl")

"""
A custom constructor for Conv2D + BatchNorm + Activation layer
    w1          : First dimension of the kernel
    w2          : Second dimension of the kernel
    input_dim   : Channel dimension of the input
    output_dim  : Filter size
    f           : For relu pass "relu" and for leaky relu pass "leaky_relu"
    alpha       : Activation constant for leaky relu, please leave as default or set to 0 for relu
    pdrop       : Dropout ratio
    padding     : Padding size
    stride      : Stride size
    dilation    : Dilation size
"""
struct ConvBn conv; bn; f; alpha; end

function ConvBn(w1::Int, w2::Int, input_dim::Int, output_dim::Int; f=nothing, alpha=0, 
    pdrop=0, bias=true, padding=0, stride=1, dilation=1, dtype=Array{Float64}, momentum=0.1)
    return ConvBn(
        Conv2D(w1, w2, input_dim, output_dim, pdrop=pdrop, dtype=dtype,
                padding=padding, stride=stride, dilation=dilation, bias=bias),
        BatchNorm(momentum=momentum),
        f, alpha
    )
end

function (cbr::ConvBn)(x; train=true)
    x_val = cbr.bn(cbr.conv(x, train=train), train=train)
    if cbr.f === nothing return x_val
    elseif cbr.f == relu || cbr.f == "relu" || cbr.f == "leaky_relu" 
        return max.(cbr.alpha .* x_val, x_val)
    else
        return cbr.f.(x_val)
    end
end

"""
Residual Layer structure with kernel sizes 1x1, 3x3, 1x1 in order. Used mostly in ResNet
Network with more than 50 layers.
"""
mutable struct Residual_1x3x1  downsample; conv_bn1; conv_bn2; conv3; bn3; end

function Residual_1x3x1(input_dim, filter_sizes; downsample=false, ds_3x3_stride=1,
                        bias=false, momentum=0.1, pdrop=0, dtype=Array{Float64})
    ds_layer = nothing
    if downsample
        ds_layer = ConvBn(1, 1, input_dim, filter_sizes[3], bias=bias, 
                            momentum=momentum, dtype=dtype, stride=ds_3x3_stride)
    end
    
    return Residual_1x3x1(
        ds_layer,
        ConvBn(1, 1, input_dim, filter_sizes[1], bias=bias, momentum=momentum, 
                dtype=dtype, pdrop=pdrop, f=relu),
        ConvBn(3, 3, filter_sizes[1], filter_sizes[2], padding=1, bias=bias, 
                momentum=momentum, dtype=dtype, pdrop=pdrop, f=relu, stride=ds_3x3_stride),
        Conv2D(1, 1, filter_sizes[2], filter_sizes[3], pdrop=pdrop, dtype=dtype, bias=bias),
        BatchNorm(momentum=momentum)
    )
end

function (r::Residual_1x3x1)(x; train=true)
    x_val = r.conv_bn1(x, train=train)
    x_val = r.conv_bn2(x_val, train=train)
    x_val = r.conv3(x_val, train=train)
    if r.downsample === nothing x_val += x
    else x_val += r.downsample(x, train=train)
    end
    x_val = relu.(r.bn3(x_val))
    return x_val
end

"""
To add layers on top of each other.
"""
mutable struct Chain; layers; end
(c::Chain)(x; train=true) = (for l in c.layers; x = l(x, train=train); end; x)

