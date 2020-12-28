include("layers.jl")
include("initializers.jl")

"""
A custom constructor for Conv2D + BatchNorm + Activation layer
    w1          : First dimension of the kernel
    w2          : Second dimension of the kernel
    input_dim   : Channel dimension of the input
    output_dim  : Filter size
    f           : For leaky relu pass "leaky_relu"
    alpha       : Activation constant for leaky relu, please leave as default or set to 0 for relu
    pdrop       : Dropout ratio
    padding     : Padding size
    stride      : Stride size
    dilation    : Dilation size
"""
mutable struct ConvBn conv; bn; f; alpha; end

function ConvBn(w1::Int, w2::Int, input_dim::Int, output_dim::Int; init=kaiming_uniform, f=nothing, alpha=0, 
    pdrop=0, bias=true, padding=0, stride=1, dilation=1, dtype=Array{Float32}, momentum=0.1)
    return ConvBn(
        Conv2D(w1, w2, input_dim, output_dim, pdrop=pdrop, dtype=dtype, init=init,
                padding=padding, stride=stride, dilation=dilation, bias=bias),
        BatchNorm(momentum=momentum, channels=output_dim, dtype=dtype),
        f, alpha
    )
end

function (cbr::ConvBn)(x; train=true)
    x_val = cbr.bn(cbr.conv(x, train=train), train=train)
    if cbr.f === nothing return x_val
    elseif cbr.f == "leaky_relu" return max.(cbr.alpha .* x_val, x_val)
    else return cbr.f.(x_val)
    end
end

"""
Residual Layer structure with kernel sizes 1x1, 3x3, 1x1 in order. Used mostly in ResNet
Network with more than 50 layers.
"""
mutable struct Residual_1x3x1  downsample; conv_bn1; conv_bn2; conv_bn3; dtype; end

function Residual_1x3x1(input_dim, filter_sizes; downsample=false, ds_3x3_stride=1, init=kaiming_uniform,
                        bias=false, pdrop=0, dtype=Array{Float32})
    ds_layer = nothing
    if downsample
        ds_layer = ConvBn(1, 1, input_dim, filter_sizes[3], bias=bias, init=init, dtype=dtype, stride=ds_3x3_stride)
    end
    
    return Residual_1x3x1(
        ds_layer,
        ConvBn(1, 1, input_dim,       filter_sizes[1], init=init, bias=bias, dtype=dtype, pdrop=pdrop, f=relu, stride=ds_3x3_stride),
        ConvBn(3, 3, filter_sizes[1], filter_sizes[2], init=init, bias=bias, dtype=dtype, pdrop=pdrop, f=relu, padding=1),
        ConvBn(1, 1, filter_sizes[2], filter_sizes[3], init=init, bias=bias, dtype=dtype, pdrop=pdrop),
        dtype
    )
end

function (r::Residual_1x3x1)(x; train=true)
    x_val = r.conv_bn1(x, train=train)
    x_val = r.conv_bn2(x_val, train=train)
    x_val = r.conv_bn3(x_val, train=train)
    if r.downsample === nothing x_val += x
    else x_val += r.downsample(x, train=train)
    end
    return relu.(x_val)
end

function load_mat_weights(block::Residual_1x3x1, conv_w, bn_mom, bn_b, bn_mult)
    idx = 1; mom_idx = 1; 
    
    if block.downsample !== nothing
        block.downsample.conv.w = Param(conv_w[idx])
        block.downsample.bn.bn_moments = bnmoments(mean=bn_mom[mom_idx], var=bn_mom[mom_idx+1])
        block.downsample.bn.bn_params = Param(vcat(vec(bn_mult[idx]), vec(bn_b[idx])))
        idx += 1; mom_idx += 2; 
    end

    block.conv_bn1.conv.w = Param(conv_w[idx])
    block.conv_bn1.bn.bn_moments = bnmoments(mean=bn_mom[mom_idx], var=bn_mom[mom_idx+1])
    block.conv_bn1.bn.bn_params = Param(vcat(vec(bn_mult[idx]), vec(bn_b[idx])))
    idx += 1; mom_idx += 2; 
    
    block.conv_bn2.conv.w = Param(conv_w[idx])
    block.conv_bn2.bn.bn_moments = bnmoments(mean=bn_mom[mom_idx], var=bn_mom[mom_idx+1])
    block.conv_bn2.bn.bn_params = Param(vcat(vec(bn_mult[idx]), vec(bn_b[idx])))
    idx += 1; mom_idx += 2; 
    
    block.conv_bn3.conv.w = Param(conv_w[idx])
    block.conv_bn3.bn.bn_moments = bnmoments(mean=bn_mom[mom_idx], var=bn_mom[mom_idx+1])
    block.conv_bn3.bn.bn_params = Param(vcat(vec(bn_mult[idx]), vec(bn_b[idx])))
    
    return block
end

"""
To add layers on top of each other.
"""
mutable struct Chain; layers; end
(c::Chain)(x; train=true) = (for l in c.layers; x = l(x, train=train); end; x)

