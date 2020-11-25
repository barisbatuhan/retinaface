include("../core/layers.jl")
include("../core/blocks.jl")

"""
Required ResNet structures are implemented there.
"""

struct ResNet50 layer1; layer2; layer3; layer4; layer5; fc; end

function ResNet50(;input_dim=3, dtype=Array{Float64}, pdrop=0, bias=false, include_top=true)
    layer1 = _InitLayer(input_dim=input_dim, dtype=dtype, pdrop=pdrop, bias=bias)
    layer2 = Chain([
        Residual_1x3x1(64, [64, 64, 256], downsample=true, dtype=dtype, pdrop=pdrop, bias=bias),
        Residual_1x3x1(256, [64, 64, 256], dtype=dtype, pdrop=pdrop, bias=bias),
        Residual_1x3x1(256, [64, 64, 256], dtype=dtype, pdrop=pdrop, bias=bias)
    ])
    layer3 = Chain([
        Residual_1x3x1(256, [128, 128, 512], downsample=true, dtype=dtype, pdrop=pdrop, 
                        ds_3x3_stride=2, bias=bias),
        Residual_1x3x1(512, [128, 128, 512], dtype=dtype, pdrop=pdrop, bias=bias),
        Residual_1x3x1(512, [128, 128, 512], dtype=dtype, pdrop=pdrop, bias=bias),
        Residual_1x3x1(512, [128, 128, 512], dtype=dtype, pdrop=pdrop, bias=bias)
    ])
    layer4 = Chain([
        Residual_1x3x1(512, [256, 256, 1024], downsample=true, dtype=dtype, pdrop=pdrop, 
                        ds_3x3_stride=2, bias=bias),
        Residual_1x3x1(1024, [256, 256, 1024], dtype=dtype, pdrop=pdrop, bias=bias),
        Residual_1x3x1(1024, [256, 256, 1024], dtype=dtype, pdrop=pdrop, bias=bias),
        Residual_1x3x1(1024, [256, 256, 1024], dtype=dtype, pdrop=pdrop, bias=bias),
        Residual_1x3x1(1024, [256, 256, 1024], dtype=dtype, pdrop=pdrop, bias=bias),
        Residual_1x3x1(1024, [256, 256, 1024], dtype=dtype, pdrop=pdrop, bias=bias)
    ])
    layer5 = Chain([
        Residual_1x3x1(1024, [512, 512, 2048], downsample=true, dtype=dtype, pdrop=pdrop, 
                        ds_3x3_stride=2, bias=bias),
        Residual_1x3x1(2048, [512, 512, 2048], dtype=dtype, pdrop=pdrop, bias=bias),
        Residual_1x3x1(2048, [512, 512, 2048], dtype=dtype, pdrop=pdrop, bias=bias),
    ])

    fc = nothing
    if include_top fc = Dense(2048, 1000) end

    return ResNet50(layer1, layer2, layer3, layer4, layer5, fc)
end

function (rn::ResNet50)(x; train=true, return_intermediate=true)
    c2 = rn.layer2(rn.layer1(x, train=true), train=true)
    c3 = rn.layer3(c2, train=true)
    c4 = rn.layer4(c3, train=true)
    c5 = rn.layer5(c4, train=true)
    if rn.fc === nothing
        if return_intermediate return c2, c3, c4, c5, nothing
        else return c5
        end
    else
        h, w, c, n = size(c5)
        # assuming image is square shaped
        p = pool(c5, window=h, mode=2)[1,1,:,:]
        res = rn.fc(p, train=true)
        if return_intermediate return c2, c3, c4, c5, res
        else return res
        end
    end
end


struct _InitLayer conv_bn; end

function _InitLayer(;input_dim=3, output_dim=64, dtype=Array{Float64}, pdrop=0, bias=false)
    return _InitLayer(
        ConvBn(7, 7, input_dim, output_dim, bias=bias, padding=3, 
                stride=2, dtype=dtype, pdrop=pdrop)
    )
end

function (il::_InitLayer)(x; train=true)
    x_val = pool(relu.(il.conv_bn(x, train=train)), window=3, stride=2, padding=1)
    return x_val
end