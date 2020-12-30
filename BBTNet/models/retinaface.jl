using JLD2
using FileIO

# Network codes
include("../backbones/resnet.jl")
include("../backbones/fpn.jl")
include("../backbones/context_module.jl")
# BBox Related Codes
include("../utils/box_processes.jl")
# Loss Functions
include("../core/losses.jl")
# Global Configuration Parameters
include("../../configs.jl")

"""
- Takes final Context Head Outputs and converts these into proposals.
- Same structure for BBox, Classifier and Landmark tasks.
- task_len is 2 for classification, 4 for bbox and 10 for landmark points.
"""
mutable struct HeadGetter 
    layers
    task_len 

    function HeadGetter(input_dim, task_len; scale_cnt=5, dtype=Array{Float32})
        layers = []
        for s in 1:scale_cnt
            push!(layers, Conv2D(1, 1, input_dim, num_anchors*task_len, dtype=dtype, bias=true))
        end
        return new(layers, task_len)
    end
end


function (hg::HeadGetter)(xs; train=true)
    proposals = []
    getter_idx = scale_cnt == 5 ? 1 : 2
    T = hg.task_len
    for (i, x) in enumerate(xs)
        proposal = hg.layers[getter_idx](x, train=train)
        getter_idx += 1
        W, H, C, N = size(proposal); A = Int(W*H*(C/T));
        
        # converting all proposals from 4D shape to 3D   
        proposal = permutedims(proposal, (3, 1, 2, 4))
        proposal = reshape(proposal, (T, A, N))
        push!(proposals, proposal)
    end
    if T == 2
        return softmax(cat(proposals..., dims=2), dims=1)
    else
        return cat(proposals..., dims=2)
    end
end

"""
Our actual model that predicts bounding boxes and landmarks.
"""
mutable struct RetinaFace
    backbone; fpn; context_module;
    cls_head1; cls_head2;
    bbox_head1; bbox_head2;
    lm_head1; lm_head2;
    dtype
end

function RetinaFace(;dtype=Array{Float32}, load_path=nothing) 
    
    if load_path !== nothing
        return load_model(load_path)
    else
        backbone = load_mat_weights(
            ResNet50(include_top=false, dtype=dtype), 
            "./weights/imagenet-resnet-50-dag.mat"
        )
        return RetinaFace(
            backbone, FPN(dtype=dtype), ContextModule(dtype=dtype), # full baseline
            HeadGetter(256, 2, dtype=dtype), HeadGetter(256, 2, dtype=dtype), 
            HeadGetter(256, 4, dtype=dtype), HeadGetter(256, 4, dtype=dtype),
            HeadGetter(256, 10, dtype=dtype), HeadGetter(256, 10, dtype=dtype),
            dtype
        )   
    end
end


# modes:
# 0 --> for getting p_vals
# 1 --> first context head forward, 
# 2 --> second context head forward, 
function (model::RetinaFace)(x; p_vals = nothing, mode=0, train=true)
    
    if p_vals === nothing
        c2, c3, c4, c5 = model.backbone(x, return_intermediate=true, train=train)   
        p_vals = model.fpn([c2, c3, c4, c5], train=train)
        p_vals = model.context_module(p_vals, train=train) 
    end
    
    class_vals = nothing; bbox_vals = nothing; landmark_vals = nothing;    
    
    if mode == 0
        return p_vals 
    
    elseif mode == 1
        class_vals = model.cls_head1(p_vals, train=train)
        bbox_vals = model.bbox_head1(p_vals, train=train)
        landmark_vals = model.lm_head1(p_vals, train=train)
    
    elseif mode == 2
        class_vals = model.cls_head2(p_vals, train=train)
        bbox_vals = model.bbox_head2(p_vals, train=train)
        landmark_vals = model.lm_head2(p_vals, train=train)
    end
    
    return class_vals, bbox_vals, landmark_vals
end

# modes:
# 0  --> baseline forward, 
# 1  --> using both context heads for forward, 
# 2  --> second context head forward, 
function (model::RetinaFace)(x, y, mode=0, train=true, weight_decay=0)
    
    p_vals = model(x, mode=0, train=train); priors = _get_priorboxes();
    cls_vals = nothing; bbox_vals = nothing; lm_vals = nothing;
    h1c_loss = 0; h1b_loss = 0; h1l_loss = 0; # first context head losses
    h2c_loss = 0; h2b_loss = 0; h2l_loss = 0; # second context head / baseline losses
    decay_loss = 0; # decay loss if decay value is bigger than 0
    
    if mode == 1
        # do the forward pass and calculate first head loss
        cls_vals1, bbox_vals1, lm_vals1 = model(x, mode=1, p_vals=p_vals, train=train)
        h1c_loss, h1b_loss, h1l_loss = get_loss(cls_vals1, bbox_vals1, lm_vals1, y, priors, mode=1)
        priors = _decode_bboxes(convert(Array{Float32}, value(bbox_vals1)), priors)
    end
    
    cls_vals2, bbox_vals2, lm_vals2 = model(x, mode=2, p_vals=p_vals, train=train)
    h2c_loss, h2b_loss, h2l_loss = get_loss(cls_vals2, bbox_vals2, lm_vals2, y, priors, mode=2) 
    
    if weight_decay > 0 # only taking weights but not biases and moments
        for p in params(model)
            if size(size(p), 1) == 4 && size(p, 4) > 1
                decay_loss += weight_decay * sum(p .* p)
            end
        end
    end
    
    loss_cls = h1c_loss + h2c_loss
    loss_bbox = h1b_loss + h2b_loss
    loss_pts = h1l_loss + h2l_loss
    
    loss = loss_cls + lambda1 * loss_bbox + lambda2 * loss_pts + decay_loss
        
    to_print = get_losses_string(loss, loss_cls, loss_bbox, loss_pts, decay_loss)
    print(to_print); open(log_dir, "a") do io write(io, to_print) end; # saving data
        
    return loss
end

# if mode is 1, then first head IOUs are taken, otherwise second head IOUs
function get_loss(cls_vals, bbox_vals, lm_vals, y, priors; mode=2) 

    loss_cls = 0; loss_lm = 0; loss_bbox = 0; loss_decay = 0; lmN = 0; bboxN = 0;
    pos_thold = mode == 1 ? head1_pos_iou : head2_pos_iou
    neg_thold = mode == 1 ? head1_neg_iou : head2_neg_iou
            
    # helper parameters for calculating losses
    T, P, N = size(cls_vals); batch_cls = convert(Array{Int64}, zeros(P, N));
    batch_gt = cat(value(bbox_vals), value(lm_vals), dims=1)
        
    for n in 1:N 
        # loop for each input in batch, all inputs may have different box counts
        if isempty(y[n]) || (y[n] == []) || (y[n] === nothing)
            continue # if the cropped image has no faces
        end 
            
        l_cls =-log.(Array(value(cls_vals)))[conf_indices[2],:,n]; gt = y[n]; prior = nothing;
        
        if length(size(priors)) > 2 prior = priors[:,:,n]
        else prior = priors
        end
        
        gt, pos_idx, neg_idx = encode_gt_and_get_indices(gt, prior, l_cls, pos_thold, neg_thold)   
            
        if pos_idx !== nothing 
            # if boxes with high enough IOU are found                
            lm_indices = getindex.(findall(gt[15,:] .>= 0))
            gt = convert(model.dtype, gt)
                
            if size(lm_indices, 1) > 0 
                # counting only the ones with landmark data 
                batch_gt[5:14,pos_idx[lm_indices],n] = gt[5:14,lm_indices]
                lmN += length(lm_indices) # check here!
            end
                
            batch_gt[1:4,pos_idx,n] = gt[1:4,:]; bboxN += length(pos_idx);          
            batch_cls[pos_idx,n] .= conf_indices[1]; batch_cls[neg_idx,n] .= conf_indices[2]; 
        end 
    end
        
    # in case no boxes are matched in the whole batch
    bboxN = bboxN == 0 ? 1 : bboxN
    lmN = lmN == 0 ? 1 : lmN
            
    # classification negative log likelihood loss
    cls_vals = reshape(cls_vals, (2, N*P))
    batch_cls = vec(batch_cls) 
    loss_cls = nll(cls_vals, batch_cls)
    if (isinf(value(loss_cls)) || isnan(value(loss_cls))) loss_cls = 0 end 
    # regression loss of the box centers, width and height
    loss_bbox = smooth_l1(abs.(batch_gt[1:4,:,:] - bbox_vals)) / bboxN
    # box center and all landmark points regression loss per point
    loss_pts = smooth_l1(abs.(batch_gt[5:end,:,:] - lm_vals)) / lmN   
    
    return loss_cls, loss_bbox, loss_pts
end

# mode 1 for 2 context heads, mode 2 for only 1 context head
function predict_model(model::RetinaFace, x; mode=2) 
    # getting predictions
    p_vals = model(x, mode=0, train=false); priors = _get_priorboxes();
    
    if mode == 1
        cls_vals1, bbox_vals1, _ = model(x, mode=1, p_vals=p_vals, train=false)
        priors = _decode_bboxes(convert(Array{Float32}, value(bbox_vals1)), priors)
    end
    
    cls_vals, bbox_vals, lm_vals = model(x, mode=2, p_vals=p_vals, train=false)
    cls_vals = Array(cls_vals); bbox_vals = Array(bbox_vals); lm_vals = Array(lm_vals);
    
    # decoding points to min and max
    bbox_vals, lm_vals = decode_points(bbox_vals, lm_vals, priors)
    bbox_vals = _to_min_max_form(bbox_vals)
               
    cls_results = []; bbox_results = []; lm_results = [];
        
    for n in 1:size(cls_vals)[end]     
        # confidence threshold check
        indices = findall(cls_vals[conf_indices[1],:,n] .>= conf_level)
        cls_result = cls_vals[:,indices,n]
        bbox_result = bbox_vals[:,indices,n]
        lm_result = lm_vals[:,indices,n]   
        # NMS check
        indices = nms(vec(cls_result[conf_indices[1],:]), bbox_result)
        cls_result = cls_result[:,indices]
        bbox_result = bbox_result[:,indices]
        lm_result = lm_result[:,indices]   
        # pushing results
        push!(cls_results, cls_result)
        push!(bbox_results, bbox_result)
        push!(lm_results, lm_result)  
    end
    
    print("[INFO] Returning results above confidence level: ", conf_level, ".\n")
    return cls_results, bbox_results, lm_results 
end

function train_model(model::RetinaFace, reader; val_data=nothing, save_dir=nothing)
    open(log_dir, "w") do io write(io, "===== TRAINING PROCESS =====\n") end;
    
    for e in start_epoch:num_epochs
        
        (imgs, boxes), state = iterate(reader)
        iter_no = 1; last_loss = 0; 
        total_batches = size(state, 1) + size(imgs)[end]; curr_batch = 0; 
        
        while state !== nothing                
            if mod(iter_no, 5) == 1
                to_print  = "\n--- Epoch: " * string(e) 
                to_print *= " & Batch: " * string(curr_batch) * "/" 
                to_print *= string(total_batches) * "\n\n"
                print(to_print)
                open(log_dir, "a") do io write(io, to_print) end;
            end
            
            if e < lr_change_epoch[1]
                momentum!(model, [(imgs, boxes, mode, true, weight_decay)], 
                    lr=lrs[1], gamma=momentum)
            elseif e < lr_change_epoch[2]
                momentum!(model, [(imgs, boxes, mode, true, weight_decay)], 
                    lr=lrs[2], gamma=momentum)
            elseif e < lr_change_epoch[3]
                momentum!(model, [(imgs, boxes, mode, true, weight_decay)], 
                    lr=lrs[3], gamma=momentum)
            else
                momentum!(model, [(imgs, boxes, mode, true, weight_decay)], 
                    lr=lrs[4], gamma=momentum)
            end
               
            if !(length(state) == 0 || state === nothing)
                (imgs, boxes), state = iterate(reader, state)
                iter_no += 1
                curr_batch += size(imgs)[end]  
            else
                if save_dir !== nothing 
                    save_model(model, save_dir * "model_" * string(e) * ".jld2")
                end
                break
            end
        end
    end
end

function load_model(file_name)
    return Knet.load(file_name, "model",)
end

function save_model(model::RetinaFace, file_name)
    Knet.save(file_name, "model", model)
end

function get_losses_string(total_loss, loss_cls, loss_bbox, loss_lm, loss_decay)
    total = string(round.(value(total_loss); digits=3))
    cls = string(round.(value(loss_cls); digits=3))
    bbox = string(round.(value(loss_bbox); digits=3))
    lm = string(round.(value(loss_lm); digits=3))
    decay = string(round.(value(loss_decay); digits=3))
    
    if length(total) < 6 total *= "0"^(6-length(total)) end
    if length(cls) < 6 cls *= "0"^(6-length(cls)) end
    if length(bbox) < 6 bbox *= "0"^(6-length(bbox)) end
    if length(lm) < 6 lm *= "0"^(6-length(lm)) end
    if length(decay) < 6 decay *= "0"^(6-length(decay)) end    
        
    to_print  = "Total Loss: " *  total * " | " 
    to_print *= "Cls Loss: "   * cls    * " | " 
    to_print *= "Box Loss: "   * bbox   * " | " 
    to_print *= "Point Loss: " * lm     * " | " 
    to_print *= "Decay: "      * decay  * "\n"
    return to_print
end
