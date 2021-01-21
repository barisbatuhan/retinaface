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
mutable struct HeadGetter layers; task_len; scale_cnt; end

function HeadGetter(input_dim, task_len; scale_cnt=5, dtype=Array{Float32})
        layers = []
        num_anchors = scale_cnt == 3 ? 2 : 3
        for s in 1:scale_cnt
            push!(layers, Conv2D(1, 1, input_dim, num_anchors*task_len, dtype=dtype, bias=true))
        end
        return HeadGetter(layers, task_len, scale_cnt)
    end


function (hg::HeadGetter)(xs; train=true)
    proposals = []
    getter_idx = hg.scale_cnt == 5 ? 1 : 2
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
    mode; num_anchors; anchor_info; log_file;
    dtype
end

function RetinaFace(;mode=1, num_anchors=3, anchor_info=lat_5_anchors, load_path=nothing, dtype=Array{Float32}) 
    laterals = num_anchors === 2 ? 3 : 5
    if load_path !== nothing
        model = load_model(load_path)
        return RetinaFace( # restoring only weights and creating new objects with added additional params
            model.backbone, 
            FPN(
                model.fpn.o6, model.fpn.o2, model.fpn.o3, model.fpn.o4, model.fpn.o5, 
                model.fpn.merge4, model.fpn.merge3, model.fpn.merge2, laterals
            ),
            ContextModule(
                model.context_module.ssh_p2, model.context_module.ssh_p3, model.context_module.ssh_p4, 
                model.context_module.ssh_p5, model.context_module.ssh_p6, laterals
            ),
            HeadGetter(model.cls_head1.layers, model.cls_head1.task_len, laterals),
            HeadGetter(model.cls_head2.layers, model.cls_head2.task_len, laterals),
            HeadGetter(model.bbox_head1.layers, model.bbox_head1.task_len, laterals),
            HeadGetter(model.bbox_head2.layers, model.bbox_head2.task_len, laterals),
            HeadGetter(model.lm_head1.layers, model.lm_head1.task_len, laterals),
            HeadGetter(model.lm_head2.layers, model.lm_head2.task_len, laterals),
            mode, num_anchors, anchor_info, nothing, dtype
        )
    else
        backbone = load_mat_weights(
            ResNet50(include_top=false, dtype=dtype), "./weights/imagenet-resnet-50-dag.mat"
        )
        return RetinaFace(
            backbone, FPN(dtype=dtype, scale_cnt=laterals), ContextModule(dtype=dtype, scale_cnt=laterals), # full baseline
            HeadGetter(256, 2, scale_cnt=laterals, dtype=dtype), HeadGetter(256, 2, scale_cnt=laterals, dtype=dtype), 
            HeadGetter(256, 4, scale_cnt=laterals, dtype=dtype), HeadGetter(256, 4, scale_cnt=laterals, dtype=dtype),
            HeadGetter(256, 10, scale_cnt=laterals, dtype=dtype), HeadGetter(256, 10, scale_cnt=laterals, dtype=dtype),
            mode, num_anchors, anchor_info, nothing, dtype
        )   
    end
end

# modes:
# 0 --> for getting p_vals
# 1 --> first context head forward, 
# 2 --> second context head forward, 
# context_module --> enables or disables the usage of context module
function (model::RetinaFace)(x; p_vals = nothing, mode=0, context_module=true, train=true)
    
    if p_vals === nothing
        c2, c3, c4, c5 = model.backbone(x, return_intermediate=true, train=train)   
        p_vals = model.fpn([c2, c3, c4, c5], train=train)
        if context_module
            p_vals = model.context_module(p_vals, train=train) 
        end
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
# 0  --> baseline until FPN + 2nd head, 
# 1  --> until context module + cascaded structure (full model), 
# 2  --> until context module + 2nd head (no cascaded structure) 
function (model::RetinaFace)(x, y, mode=0, train=true, weight_decay=0)
    
    use_context = mode == 0 ? false : true
    p_vals = model(x, mode=0, train=train, context_module=use_context) 
    priors = _get_priorboxes(model.num_anchors, model.anchor_info, size(x, 1));
    
    cls_vals = nothing; bbox_vals = nothing; lm_vals = nothing;
    h1c_loss = 0; h1b_loss = 0; h1l_loss = 0; # first context head losses
    h2c_loss = 0; h2b_loss = 0; h2l_loss = 0; # second context head / baseline losses
    decay_loss = 0; # decay loss if decay value is bigger than 0
    
    if model.mode == 1
        # do the forward pass and calculate first head loss
        cls_vals1, bbox_vals1, lm_vals1 = model(x, mode=1, p_vals=p_vals, context_module=use_context, train=train)
        h1c_loss, h1b_loss, h1l_loss = get_loss(cls_vals1, bbox_vals1, lm_vals1, y, priors, mode=1, dtype=model.dtype)
        priors = _decode_bboxes(convert(Array{Float32}, value(bbox_vals1)), priors)
    end
    
    cls_vals2, bbox_vals2, lm_vals2 = model(x, mode=2, p_vals=p_vals, context_module=use_context, train=train)
    h2c_loss, h2b_loss, h2l_loss = get_loss(cls_vals2, bbox_vals2, lm_vals2, y, priors, mode=2, dtype=model.dtype) 
    
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
    print(to_print); 
    if model.log_file !== nothing
        open(model.log_file, "a") do io write(io, to_print) end; # saving data
    end
        
    return loss
end

# if mode is 1, then first head IOUs are taken, otherwise second head IOUs
function get_loss(cls_vals, bbox_vals, lm_vals, y, priors; mode=2, dtype=Array{Float32}) 

    loss_cls = 0; loss_lm = 0; loss_bbox = 0; loss_decay = 0; lmN = 0; bboxN = 0;
    pos_thold = mode == 1 ? head1_pos_iou : head2_pos_iou
    neg_thold = mode == 1 ? head1_neg_iou : head2_neg_iou
            
    # helper parameters for calculating losses
    T, P, N = size(cls_vals); batch_cls = convert(Array{Int64}, zeros(2, P, N));
    batch_gt = cat(value(bbox_vals), value(lm_vals), dims=1)
        
    for n in 1:N # loop for each input in batch, all inputs may have different box counts
        if isempty(y[n]) || (y[n] == []) || (y[n] === nothing)
            continue # if the cropped image has no faces
        end 
            
        l_cls =Array(value(cls_vals))[1,:,n]; gt = y[n];
        
        prior = priors
        if length(size(priors)) > 2 
            prior = priors[:,:,n] # retuired for mode 1 
        end
        
        gt, pos_idx, neg_idx = encode_gt_and_get_indices(gt, prior, l_cls, pos_thold, neg_thold)   
            
        if pos_idx !== nothing 
            # if boxes with high enough IOU are found                
            lm_indices = getindex.(findall(gt[15,:] .>= 0)); gt = convert(dtype, gt);
                
            if size(lm_indices, 1) > 0 
                # counting only the ones with landmark data 
                batch_gt[5:14,pos_idx[lm_indices],n] = gt[5:14,lm_indices]
                lmN += length(lm_indices)
            end
                
            batch_gt[1:4,pos_idx,n] = gt[1:4,:]; bboxN += length(pos_idx);    
            batch_cls[1,neg_idx,n] .= 1; batch_cls[2,pos_idx,n] .= 2;       
        end 
    end
        
    # in case no boxes are matched in the whole batch
    bboxN = bboxN == 0 ? 1 : bboxN
    lmN = lmN == 0 ? 1 : lmN
            
    # classification negative log likelihood loss
    cls_vals = reshape(cls_vals, (2, N*P))
    loss_cls_neg = nll(cls_vals, vec(batch_cls[1,:,:]))
    loss_cls_pos = nll(cls_vals, vec(batch_cls[2,:,:]))
    loss_cls = (ohem_ratio * loss_cls_pos + loss_cls_neg) / (ohem_ratio+1)
    if (isinf(value(loss_cls)) || isnan(value(loss_cls))) loss_cls = 0 end 
    
    # regression loss of the box centers, width and height
    loss_bbox = smooth_l1(abs.(batch_gt[1:4,:,:] - bbox_vals)) / bboxN
    
    # box center and all landmark points regression loss per point
    loss_pts = smooth_l1(abs.(batch_gt[5:end,:,:] - lm_vals)) / lmN   
    
    return loss_cls, loss_bbox, loss_pts
end

# modes:
# 0  --> baseline until FPN + 2nd head, 
# 1  --> until context module + cascaded structure (full model), 
# 2  --> until context module + 2nd head (no cascaded structure) 
# set filter to false for not making any confidence score and NMS check (for evaluation) 
function predict_image(model::RetinaFace, x; mode=1, filter=true, verbose=true) 
    use_context = mode == 0 ? false : true
    img_size = size(x, 1)
    # getting predictions
    p_vals = model(x, mode=0, context_module=use_context, train=false) 
    priors = _get_priorboxes(model.num_anchors, model.anchor_info, img_size)
    
    if mode == 1
        _, bbox_vals1, _ = model(x, mode=1, p_vals=p_vals, context_module=use_context, train=false)
        priors = _decode_bboxes(convert(Array{Float32}, value(bbox_vals1)), priors)[:,:,1]
    end
    
    cls_vals, bbox_vals, lm_vals = model(x, mode=2, p_vals=p_vals, context_module=use_context, train=false)
    cls_vals = Array(cls_vals); bbox_vals = Array(bbox_vals); lm_vals = Array(lm_vals);
    
    # decoding points to min and max
    bbox_vals, lm_vals = decode_points(bbox_vals, lm_vals, priors)
    bbox_vals = _to_min_max_form(bbox_vals)
    
    if !filter
        return cls_vals[:,:,1], bbox_vals[:,:,1], lm_vals[:,:,1]
    else
        indices = findall(cls_vals[2,:,1] .>= conf_level)
        cls_result = cls_vals[:,indices,1]
        bbox_result = bbox_vals[:,indices,1]
        lm_result = lm_vals[:,indices,1]   
        if verbose
            print("[INFO] Passed Confidence Check: ", size(indices, 1), "\n")
        end
        indices = nms(vec(cls_result[2,:]), bbox_result)
        cls_result = cls_result[:,indices]
        bbox_result = bbox_result[:,indices]
        lm_result = lm_result[:,indices]   
        if verbose
            print("[INFO] Passed NMS Check: ", size(indices, 1),"\n")
            print("[INFO] Returning results above confidence level: ", conf_level, "\n")
        end
        return cls_result, bbox_result, lm_result
    end
end

function train_model(model::RetinaFace, reader; val_data=nothing, save_dir=nothing, start_epoch=1, log_file=nothing)
    
    if log_file !== nothing
        model.log_file = log_file
        open(log_file, "w") do io write(io, "===== TRAINING PROCESS =====\n") end;
    end
    
    # Adjusting LR for each step
    lrs_per_epoch = zeros(num_epochs); lr_change = 0;
    for e in 1:num_epochs
        if e == 1
            lrs_per_epoch[e] = lrs[1]; lr_change = (lrs[2] - lrs[1]) / lr_change_epoch[1];
        elseif e == lr_change_epoch[1]
            lrs_per_epoch[e] = lrs[2]; lr_change = (lrs[3] - lrs[2]) / (lr_change_epoch[2] - lr_change_epoch[1]);
        elseif e == lr_change_epoch[2]
            lrs_per_epoch[e] = lrs[3]; lr_change = (lrs[4] - lrs[3]) / (lr_change_epoch[3] - lr_change_epoch[2]); 
        elseif e == lr_change_epoch[3]
            lrs_per_epoch[e] = lrs[4]; lr_change = 0;
        else
            lrs_per_epoch[e] = lrs_per_epoch[e - 1] + lr_change  
        end
    end
    
    for e in start_epoch:num_epochs
        (imgs, boxes), state = iterate(reader)
        iter_no = 1; last_loss = 0; total_batches = size(state, 1) + size(imgs)[end]; 
        curr_batch = 0; curr_lr = lrs_per_epoch[e]
        
        while state !== nothing && imgs !== nothing      
            if mod(iter_no, 5) == 1 # prints per 5 batches
                to_print  = "\n--- Epoch: " * string(e) * " & LR: " * string(round(curr_lr; digits=4))
                to_print *= " & Batch: " * string(curr_batch) * "/" * string(total_batches) * "\n\n"
                print(to_print)
                if model.log_file !== nothing 
                    open(model.log_file, "a") do io write(io, to_print) end;
                end
            end

            # Updating the model
            momentum!(model, [(imgs, boxes, model.mode, true, weight_decay)], lr=curr_lr, gamma=momentum)
            (imgs, boxes), state = iterate(reader, state)
            iter_no += 1; 
            if imgs !== nothing curr_batch += size(imgs)[end] end
        end
        if save_dir !== nothing
            to_save = save_dir * "model_"
            if model.mode == 0
                to_save *= "mode0_epoch" * string(e) * ".jld2"
            elseif model.mode == 1
                to_save *= "mode1_epoch" * string(e) * ".jld2"
            elseif model.mode == 2
                to_save *= "mode2_epoch" * string(e) * ".jld2"
            else
                to_save *= "unknownmode_epoch" * string(e) * ".jld2"
            end
            save_model(model, to_save)
        end
    end
end

function load_model(file_name)
    return Knet.load(file_name, "model")
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
