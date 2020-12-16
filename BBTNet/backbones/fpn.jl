include("../core/blocks.jl")

"""
Upsampling and Pyramid Network for the RetinaFace. 
"""
struct FPN o6; o2; o3; o4; o5; merge4; merge3; merge2; end

function FPN(;in_size=2048, dtype=Array{Float64})
    return FPN(
        ConvBn(3, 3, in_size, 256, init=xavier, stride=2, bias=false, padding=1, dtype=dtype),
        ConvBn(1, 1, 256, 256, f=relu, bias=false, dtype=dtype),
        ConvBn(1, 1, 512, 256, f=relu, bias=false, dtype=dtype),
        ConvBn(1, 1, 1024, 256, f=relu, bias=false, dtype=dtype),
        ConvBn(1, 1, 2048, 256, f=relu, bias=false, dtype=dtype),
        ConvBn(3, 3, 256, 256, f=relu, padding=1, bias=false, dtype=dtype),
        ConvBn(3, 3, 256, 256, f=relu, padding=1, bias=false, dtype=dtype),
        ConvBn(3, 3, 256, 256, f=relu, padding=1, bias=false, dtype=dtype),
    )
end

function (fpn::FPN)(xs; train=true)
    c2, c3, c4, c5 = xs
    # extra conv layer
    p6 = fpn.o6(c5, train=train)
    # refucing the dim. to 256 and upsampling
    p5 = fpn.o5(c5, train=train)
    p4 = fpn.o4(c4, train=train) + unpool(p5)
    p3 = fpn.o3(c3, train=train) + unpool(p4)
    p2 = fpn.o2(c2, train=train) + unpool(p3)
    # final 3x3 conv layers
    p4 = fpn.merge4(p4, train=train)
    p3 = fpn.merge3(p3, train=train)
    p2 = fpn.merge2(p2, train=train)
    return [p2, p3, p4, p5, p6]
end