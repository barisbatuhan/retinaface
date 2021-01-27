using Knet, JLD

include("retinaface.jl")
include("../backbones/resnet.jl")
include("../backbones/fpn.jl")
include("../backbones/context_module.jl")
include("../backbones/ssh.jl")


function load_pth_model(model::RetinaFace, path; dtype=Array{Float32}, load_heads=true)
    c = JLD.jldopen(path, "r") do file
        read(file, "data")
    end
    data = Dict()
    for k in keys(c)
        if !isempty(k) data[k] = c[k] end
    end
    return set_pth_data(model, data; dtype=dtype, load_heads=load_heads)
end

function set_pth_data(model::RetinaFace, data; dtype=Array{Float32}, load_heads=true)
    conv_w, conv_b, bn_mom, bn_b, bn_mult = get_pth_resnet_params(data, dtype=dtype)
    model.backbone = load_mat_weights(model.backbone, nothing; pre_weights=[conv_w, conv_b, nothing, nothing, bn_mom, bn_b, bn_mult])
    model = set_pth_fpn_data(model, data, dtype=dtype)
    model = set_pth_context_module_data(model, data, dtype=dtype) 
    if load_heads
        model = set_pth_head_getters_data(model, data, dtype=dtype)
    end
    
    for p in params(model)
        p = Param(convert(dtype, value(p)))
    end
    
    return model
end

function get_pth_resnet_params(data; dtype=Array{Float32})
    conv_w = []; conv_b = convert(dtype, zeros(1, 1, size(data["module.body.conv1.weight"], 1), 1)); 
    bn_mom = []; bn_b = []; bn_mult = [];  
    
    push!(conv_w, convert(dtype, reverse(reverse(permutedims(data["module.body.conv1.weight"], (4, 3, 2, 1)), dims=1), dims=2)))
    push!(bn_mom, convert(dtype, reshape(data["module.body.bn1.running_mean"], (1, 1, size(data["module.body.bn1.running_mean"], 1), 1))))
    push!(bn_mom, convert(dtype, reshape(data["module.body.bn1.running_var"], (1, 1, size(data["module.body.bn1.running_var"], 1), 1))))
    push!(bn_mult, convert(dtype, vec(data["module.body.bn1.weight"])))
    push!(bn_b, convert(dtype, vec(data["module.body.bn1.bias"])))
    
    stages = ["layer1", "layer2", "layer3", "layer4"]
    
    for stage in stages
        ws, moms, bs, mults = get_pth_block_data(data, stage, dtype=dtype)
        conv_w = vcat(conv_w, ws); 
        bn_mom = vcat(bn_mom, moms);  bn_mult = vcat(bn_mult, mults);  bn_b = vcat(bn_b, bs);
    end
    return conv_w, conv_b, bn_mom, bn_b, bn_mult
end

function set_pth_fpn_data(model::RetinaFace, data; dtype=Array{Float32})
    # ResNet-to-FPN connections
    model.fpn.o5.conv.w = Param(convert(dtype, reverse(reverse(permutedims(data["module.fpn.output3.0.weight"], (4, 3, 2, 1)), dims=1), dims=2)))
    model.fpn.o5.bn.bn_params = Param(convert(dtype, vcat(
            vec(data["module.fpn.output3.1.weight"]),
            vec(data["module.fpn.output3.1.bias"]))))
    model.fpn.o5.bn.bn_moments = bnmoments(
        mean=convert(dtype, data["module.fpn.output3.1.running_mean"]),
        var=convert(dtype, data["module.fpn.output3.1.running_var"]))
    
    model.fpn.o4.conv.w = Param(convert(dtype, reverse(reverse(permutedims(data["module.fpn.output2.0.weight"], (4, 3, 2, 1)), dims=1), dims=2)))
    model.fpn.o4.bn.bn_params = Param(convert(dtype, vcat(
            vec(data["module.fpn.output2.1.weight"]),
            vec(data["module.fpn.output2.1.bias"]))))
    model.fpn.o4.bn.bn_moments = bnmoments(
        mean=convert(dtype, data["module.fpn.output2.1.running_mean"]), 
        var=convert(dtype, data["module.fpn.output2.1.running_var"]))
    
    model.fpn.o3.conv.w = Param(convert(dtype, reverse(reverse(permutedims(data["module.fpn.output1.0.weight"], (4, 3, 2, 1)), dims=1), dims=2)))
    model.fpn.o3.bn.bn_params = Param(convert(dtype, vcat(
            vec(data["module.fpn.output1.1.weight"]), 
            vec(data["module.fpn.output1.1.bias"]))))
    model.fpn.o3.bn.bn_moments = bnmoments(
        mean=convert(dtype, data["module.fpn.output1.1.running_mean"]), 
        var=convert(dtype, data["module.fpn.output1.1.running_var"]))
    
    # Merge connections
    model.fpn.merge4.conv.w = Param(convert(dtype, reverse(reverse(permutedims(data["module.fpn.merge2.0.weight"], (4, 3, 2, 1)), dims=1), dims=2)))
    model.fpn.merge4.bn.bn_params = Param(convert(dtype, vcat(
            vec(data["module.fpn.merge2.1.weight"]), 
            vec(data["module.fpn.merge2.1.bias"]))))
    model.fpn.merge4.bn.bn_moments = bnmoments(
        mean=convert(dtype, data["module.fpn.merge2.1.running_mean"]), 
        var=convert(dtype, data["module.fpn.merge2.1.running_var"]))
    
    model.fpn.merge3.conv.w = Param(convert(dtype, reverse(reverse(permutedims(data["module.fpn.merge1.0.weight"], (4, 3, 2, 1)), dims=1), dims=2)))
    model.fpn.merge3.bn.bn_params = Param(convert(dtype, vcat(
            vec(data["module.fpn.merge1.1.weight"]),
            vec(data["module.fpn.merge1.1.bias"]))))
    model.fpn.merge3.bn.bn_moments = bnmoments(
        mean=convert(dtype, data["module.fpn.merge1.1.running_mean"]), 
        var=convert(dtype, data["module.fpn.merge1.1.running_var"]))
    
    return model
end

function set_pth_context_module_data(model::RetinaFace, data; dtype=Array{Float32})
    model.context_module.ssh_p3 = set_pth_ssh_data(model.context_module.ssh_p3, "module.ssh1", data, dtype=dtype)
    model.context_module.ssh_p4 = set_pth_ssh_data(model.context_module.ssh_p4, "module.ssh2", data, dtype=dtype)
    model.context_module.ssh_p5 = set_pth_ssh_data(model.context_module.ssh_p5, "module.ssh3", data, dtype=dtype)
    return model
end

function set_pth_head_getters_data(model::RetinaFace, data; dtype=Array{Float32})
    model.cls_head2 = set_pth_head_data(model.cls_head2, "module.ClassHead", data, dtype=dtype)
    model.bbox_head2 = set_pth_head_data(model.bbox_head2, "module.BboxHead", data, dtype=dtype)
    model.lm_head2 = set_pth_head_data(model.lm_head2, "module.LandmarkHead", data, dtype=dtype)
    return model
end

function set_pth_head_data(model::HeadGetter, start_txt, data; dtype=Array{Float32})
    model.layers[1].w = Param(convert(dtype, reverse(reverse(permutedims(data[start_txt * ".0.conv1x1.weight"], (4, 3, 2, 1)), dims=1), dims=2)))
    model.layers[1].b = Param(convert(dtype, 
        reshape(data[start_txt * ".0.conv1x1.bias"], (1, 1, size(data[start_txt * ".0.conv1x1.bias"], 1), 1))))
    
    model.layers[2].w = Param(convert(dtype, reverse(reverse(permutedims(data[start_txt * ".1.conv1x1.weight"], (4, 3, 2, 1)), dims=1), dims=2)))
    model.layers[2].b = Param(convert(dtype, 
        reshape(data[start_txt * ".1.conv1x1.bias"], (1, 1, size(data[start_txt * ".1.conv1x1.bias"], 1), 1))))
    
    model.layers[3].w = Param(convert(dtype, reverse(reverse(permutedims(data[start_txt * ".2.conv1x1.weight"], (4, 3, 2, 1)), dims=1), dims=2)))
    model.layers[3].b = Param(convert(dtype, 
        reshape(data[start_txt * ".2.conv1x1.bias"], (1, 1, size(data[start_txt * ".2.conv1x1.bias"], 1), 1))))
    
    if start_txt == "module.ClassHead"
        model.task_len = 2
    elseif start_txt == "module.BboxHead"
        model.task_len = 4
    else
        model.task_len = 10
    end
    
    return model
end

function set_pth_ssh_data(model::SSH, start_txt, data; dtype=Array{Float32})
    model.conv128.conv.w = Param(convert(dtype, reverse(reverse(permutedims(data[start_txt * ".conv3X3.0.weight"], (4, 3, 2, 1)), dims=1), dims=2)))
    model.conv128.bn.bn_params = Param(convert(dtype, vcat(
        vec(data[start_txt * ".conv3X3.1.weight"]), vec(data[start_txt * ".conv3X3.1.bias"]))))
    model.conv128.bn.bn_moments = bnmoments(
        mean=convert(dtype, data[start_txt * ".conv3X3.1.running_mean"]), 
        var=convert(dtype, data[start_txt * ".conv3X3.1.running_var"]))
    
    model.conv64_1_1.conv.w = Param(convert(dtype, reverse(reverse(permutedims(data[start_txt * ".conv5X5_1.0.weight"], (4, 3, 2, 1)), dims=1), dims=2)))
    model.conv64_1_1.bn.bn_params = Param(convert(dtype, vcat(
        vec(data[start_txt * ".conv5X5_1.1.weight"]), vec(data[start_txt * ".conv5X5_1.1.bias"]))))
    model.conv64_1_1.bn.bn_moments = bnmoments(
        mean=convert(dtype, data[start_txt * ".conv5X5_1.1.running_mean"]), 
        var=convert(dtype, data[start_txt * ".conv5X5_1.1.running_var"]))
    
    model.conv64_1_2.conv.w = Param(convert(dtype, reverse(reverse(permutedims(data[start_txt * ".conv5X5_2.0.weight"], (4, 3, 2, 1)), dims=1), dims=2)))
    model.conv64_1_2.bn.bn_params = Param(convert(dtype, vcat(
        vec(data[start_txt * ".conv5X5_2.1.weight"]), vec(data[start_txt * ".conv5X5_2.1.bias"]))))
    model.conv64_1_2.bn.bn_moments = bnmoments(
        mean=convert(dtype, data[start_txt * ".conv5X5_2.1.running_mean"]), 
        var=convert(dtype, data[start_txt * ".conv5X5_2.1.running_var"]))
    
    model.conv64_2_1.conv.w = Param(convert(dtype, reverse(reverse(permutedims(data[start_txt * ".conv7X7_2.0.weight"], (4, 3, 2, 1)), dims=1), dims=2)))
    model.conv64_2_1.bn.bn_params = Param(convert(dtype, vcat(
        vec(data[start_txt * ".conv7X7_2.1.weight"]), vec(data[start_txt * ".conv7X7_2.1.bias"]))))
    model.conv64_2_1.bn.bn_moments = bnmoments(
        mean=convert(dtype, data[start_txt * ".conv7X7_2.1.running_mean"]), 
        var=convert(dtype, data[start_txt * ".conv7X7_2.1.running_var"]))
    
    model.conv64_2_2.conv.w = Param(convert(dtype, reverse(reverse(permutedims(data[start_txt * ".conv7x7_3.0.weight"], (4, 3, 2, 1)), dims=1), dims=2)))
    model.conv64_2_2.bn.bn_params = Param(convert(dtype, vcat(
        vec(data[start_txt * ".conv7x7_3.1.weight"]), vec(data[start_txt * ".conv7x7_3.1.bias"]))))
    model.conv64_2_2.bn.bn_moments = bnmoments(
        mean=convert(dtype, data[start_txt * ".conv7x7_3.1.running_mean"]), 
        var=convert(dtype, data[start_txt * ".conv7x7_3.1.running_var"]))
    
    return model
end

function get_pth_block_data(data, stagename; dtype=Array{Float32})
    conv_w = []; bn_mom = []; bn_b = []; bn_mult = [];
    conv_ds = nothing; ds_bn_b = nothing; ds_bn_mult = nothing; ds_bn_moms = [];
    
    kvals = []
    for k in keys(data)
        if !isempty(k)
            push!(kvals, string(k))
        end
    end
    
    for k in sort(kvals)
        if typeof(data[k]) == Int64
            continue
        end
        
        val = data[k]
        # print(k, " --> ", size(val), "\n")
        
        if startswith(k, "module.body." * stagename) && occursin("conv", k) && endswith(k, "weight")
            # residual weight
            push!(conv_w, convert(dtype, reverse(reverse(permutedims(val, (4, 3, 2, 1)), dims=1), dims=2)))
        elseif startswith(k, "module.body." * stagename) && endswith(k, "downsample.0.weight")
            # downsample weight
            conv_ds = convert(dtype, reverse(reverse(permutedims(val, (4, 3, 2, 1)), dims=1), dims=2))
        else
            # downsample parameters
            val = convert(dtype, val)
            if startswith(k, "module.body." * stagename) && endswith(k, "downsample.1.bias")
                ds_bn_b = vec(val)
            elseif startswith(k, "module.body." * stagename) && endswith(k, "downsample.1.weight")
                ds_bn_mult = vec(val)
            elseif startswith(k, "module.body." * stagename) && endswith(k, "downsample.1.running_mean")
                push!(ds_bn_moms, reshape(val, (1, 1, size(val, 1), 1)))
            elseif startswith(k, "module.body." * stagename) && endswith(k, "downsample.1.running_var")
                push!(ds_bn_moms, reshape(val, (1, 1, size(val, 1), 1)))     
            # residual parameters
            elseif startswith(k, "module.body." * stagename) && endswith(k, "bias")
                push!(bn_b, vec(val))
            elseif startswith(k, "module.body." * stagename) && endswith(k, "weight")
                push!(bn_mult, vec(val))  
            elseif startswith(k, "module.body." * stagename) && occursin("running_", k)
                push!(bn_mom, reshape(val, (1, 1, size(val, 1), 1)))
            end
        end
    end
    
    conv_w = [conv_ds, conv_w...]
    bn_mom = [ds_bn_moms..., bn_mom...]
    bn_b = [ds_bn_b, bn_b...]
    bn_mult = [ds_bn_mult, bn_mult...]
    return conv_w, bn_mom, bn_b, bn_mult
end
