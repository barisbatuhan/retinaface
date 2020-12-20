using ProgressBars, Printf
using JLD2
using FileIO

# Network codes
include("../backbones/resnet.jl")
include("../backbones/fpn.jl")
include("../backbones/ssh.jl")
# BBox Related Codes
include("../utils/box_processes.jl")
# Loss Functions
include("../core/losses.jl")
# Global Configuration Parameters
include("../../configs.jl")

"""
Takes final Context Head Outputs and converts these into proposals.
Same structure for BBox, Classifier and Landmark tasks.
task_len is 2 for classification, 4 for bbox and 10 for landmark
"""
struct HeadGetter layers; task_len; end

function HeadGetter(input_dim, num_anchors, task_len; scale_cnt=5, dtype=Array{Float64})
    layers = []
    for s in 1:scale_cnt
        push!(layers, Conv2D(1, 1, input_dim, 
                num_anchors*task_len, dtype=dtype, bias=false))
    end
    return HeadGetter(layers, task_len)
end

function (hg::HeadGetter)(xs; train=true)
    proposals = []
    for (i, x) in enumerate(xs)
        proposal = hg.layers[i](x, train=train)
        batch_size = size(proposal)[end]
        # flattening all of the proposals
        num_proposals = Int(floor(prod(size(proposal)) / (hg.task_len * batch_size)))
        proposal = reshape(proposal, (batch_size, num_proposals, hg.task_len))
        push!(proposals, proposal)
    end
    if hg.task_len == 2 && train == false 
        return softmax(cat(proposals..., dims=2), dims=3)
    else 
        return cat(proposals..., dims=2)
    end
end

"""
Our actual model that predicts bounding boxes and landmarks.
"""
struct RetinaFace
    backbone
    fpn
    head_module1
    head_module2
    class_conv1
    class_conv2
    bbox_conv1
    bbox_conv2
    landmark_conv1
    landmark_conv2
    dtype
end

function RetinaFace(;dtype=Array{Float64}, load_path=nothing) 
    
    if load_path !== nothing
        return load_model(load_path)
    else
        backbone = load_mat_weights(ResNet50(include_top=false, dtype=dtype), "./weights/imagenet-resnet-50-dag.mat")
        return RetinaFace(
            backbone,
            FPN(dtype=dtype), SSH(dtype=dtype), SSH(dtype=dtype),
            HeadGetter(256, num_anchors, 2, dtype=dtype), HeadGetter(256, num_anchors, 2, dtype=dtype),
            HeadGetter(256, num_anchors, 4, dtype=dtype), HeadGetter(256, num_anchors, 4, dtype=dtype),
            HeadGetter(256, num_anchors, 10, dtype=dtype), HeadGetter(256, num_anchors, 10, dtype=dtype),
            dtype
        )   
    end
end

# mode 1 means first context head, 2 means second context head, 0 means no context head
function (model::RetinaFace)(x, y=nothing, mode=0, train=true, weight_decay=0)
    # first processes
    c2, c3, c4, c5 = model.backbone(x, return_intermediate=true, train=train)
    p_vals = model.fpn([c2, c3, c4, c5], train=train)
    
    class_vals = nothing; bbox_vals = nothing; landmark_vals = nothing; 
    if mode == 1
        # 1st context head module
        idx = 1
        for idx in 1:size(p_vals)[1]
            p_vals[idx] = model.head_module1(p_vals[idx], train=train)
        end
        class_vals = model.class_conv1(p_vals, train=train)
        bbox_vals = model.bbox_conv1(p_vals, train=train)
        landmark_vals = model.landmark_conv1(p_vals, train=train)
    
    else
        # 2nd context head module
        if mode == 2
            for idx in 1:size(p_vals)[1]
                p_vals[idx] = model.head_module2(p_vals[idx], train=train)
            end
        end
        class_vals = model.class_conv2(p_vals, train=train)
        bbox_vals = model.bbox_conv2(p_vals, train=train)
        landmark_vals = model.landmark_conv2(p_vals, train=train)
    end

    if y === nothing && train == false
        # for predicting, the founded boxes should be decoded to their real values
        class_vals = Array(class_vals)
        bbox_vals = Array(bbox_vals); landmark_vals = Array(landmark_vals);
        bbox_vals, landmark_vals = decode_points(bbox_vals, landmark_vals)
        
        N = size(class_vals)[1]
        cl_results = []; bbox_results = []; landmark_results = []
        
        for n in 1:N 
            
            bbox_vals[findall(bbox_vals .< 0)] .= 0
            bbox_vals[findall(bbox_vals .> img_size)] .= img_size
            
            # confidence threshold check
            indices = findall(class_vals[n,:,1] .>= conf_level)
            cl_result = class_vals[n, indices, :]
            bbox_result = bbox_vals[n, indices, :]
            landmark_result = landmark_vals[n, indices, :]
        
            # NMS check
            indices = nms(cl_result, bbox_result)
            cl_result = cl_result[indices,:]
            bbox_result = bbox_result[indices,:]
            landmark_result = landmark_result[indices,:]
            # pushing results
            push!(cl_results, cl_result)
            push!(bbox_results, bbox_result)
            push!(landmark_results, landmark_result)
        end  
        print("[INFO] Returning results above confidence level: ", conf_level, ".\n")
        return  cl_results, bbox_results, landmark_results 
    
    else
        # for training, loss will be calculated and returned
        loss_cls = 0; loss_lm = 0; loss_bbox = 0; loss_decay = 0;
        lmN = 0; bboxN = 0;
        pos_thold = mode == 1 ? head1_pos_iou : head2_pos_iou
        neg_thold = mode == 1 ? head1_neg_iou : head2_neg_iou
        
        N = size(class_vals, 1)
        batch_gt = cat(value(bbox_vals), value(landmark_vals), dims=3)
        batch_cls = convert(Array{Int64}, zeros(N, size(class_vals, 2)))
        
        for n in 1:N # loop for each input in batch, all inputs may have different box counts
            if isempty(y[n]) || (y[n] == []) || (y[n] === nothing)
                # if the cropped image has no faces
                continue 
            end 
            
            gt, pos_indices, neg_indices = encode_gt_and_get_indices(
                permutedims(y[n],(2, 1)), pos_thold, neg_thold)   
            
            if pos_indices !== nothing 
                lm_indices = findall(gt[:,15] .>= 0)
                if size(lm_indices, 1) > 0 
                    # getting indices where landmark points are available     
                    lm_indices = getindex.(lm_indices)
                    lmN += size(lm_indices, 1) # * 5
                end
                # if boxes with high enough IOU are found
                gt = convert(model.dtype, gt)
                batch_gt[n,pos_indices,1:4] = gt[:,1:4] 
                if lm_indices !== nothing 
                    # counting only the ones with landmark data
                    batch_gt[n,pos_indices[lm_indices],5:14] = gt[lm_indices,5:14]
                end
                batch_cls[n, pos_indices] .= 1
                batch_cls[n, neg_indices] .= 2
                bboxN += size(pos_indices, 1) # * 2
            end 
        end
        
        bboxN = bboxN == 0 ? 1 : bboxN
        lmN = lmN == 0 ? 1 : lmN
            
        # classification negative log likelihood loss
        class_vals = reshape(class_vals, (2, prod(size(class_vals)[1:2])))
        batch_cls = reshape(batch_cls, (1, prod(size(batch_cls))))
        loss_cls = nll(class_vals, batch_cls)
        # box regression loss
        loss_bbox = smooth_l1(abs.(batch_gt[:,:,1:4] .- bbox_vals)) / bboxN
        # landmark regression loss
        loss_lm = smooth_l1(abs.(batch_gt[:,:,5:end] .- landmark_vals)) / lmN
            
        # weight decay
        if weight_decay > 0
            for p in params(model)
                if size(size(p), 1) == 4 && size(p, 4) > 1
                    # only taking weights but not biases and moments
                    loss_decay += weight_decay * sum(p.^2)
                end
            end
        end
        
        loss_cls = loss_cls === NaN ? 0 : loss_cls  
        total_loss = loss_cls + lambda1 * loss_bbox + lambda2 * loss_lm - loss_decay
        
        to_print = get_losses_string(
            total_loss, loss_cls, loss_bbox, loss_lm, loss_decay)
        
        if total_loss > 0
            print(to_print)
            open(log_dir, "a") do io write(io, to_print) end;
        end
        return total_loss
    end
end

function train_model(model::RetinaFace, data_reader; val_data=nothing, save_dir=nothing)
    print("\n--> TRAINING PROCESS:\n\n")
    open(log_dir, "w") do io write(io, "===== TRAINING PROCESS =====\n\n") end;

    for e in start_epoch:num_epochs
        (imgs, boxes), state = iterate(data_reader)
        iter_no = 1
        last_loss = 0
        total_batches = size(state, 1) + size(imgs)[end]
        curr_batch = 0
        
        while state !== nothing 
            # for running the same batch over and over
            # (imgs, boxes), _ = iterate(data_reader, state)         
            
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
                (imgs, boxes), state = iterate(data_reader, state)
                iter_no += 1
                curr_batch += size(imgs)[end]
                if save_dir !== nothing && mod(iter_no, 644) == 0
                    save_model(model, save_dir * "model_epoch" * string(e) * "_iter" * string(curr_batch) * ".jld2")    
                end   
            else
                if save_dir !== nothing 
                    save_model(model, save_dir * "model_epoch" * string(e) * ".jld2")
                end
                break
            end
        end
        
        # Evaluate both training and val data after each epoch.
#         train_loss = evaluate_model(model, data_reader)
#         print("\nEpoch: ", e, " ---> Train Loss: ", train_loss)
#         if val_data !== nothing
#             val_loss = evaluate_model(model, val_data)
#             push!(loss_history, (train_loss, val_loss))
#             print(" || Validation Loss: ", val_loss)
#         else
#             push!(loss_history, train_loss)
#         end
#         print("\n")
    end
end

function evaluate_model(model::RetinaFace, data_reader)
    (imgs, boxes), state = iterate(data_reader)
    num_iters = 0
    loss_val = 0.0
    while state !== nothing
        loss_val += model(imgs, boxes, mode, false, 0)
        num_iters += 1
        (imgs, boxes), state = iterate(data_reader)
    end
    return loss_val / num_iters
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
    
    if length(total) < 8 total *= "0"^(8-length(total)) end
    if length(cls) < 6 cls *= "0"^(6-length(cls)) end
    if length(bbox) < 8 bbox *= "0"^(8-length(bbox)) end
    if length(lm) < 8 lm *= "0"^(8-length(lm)) end
    if length(decay) < 6 decay *= "0"^(6-length(decay)) end    
        
    to_print  = "Total Loss: " *  total * " | " 
    to_print *= "Cls Loss: "   * cls    * " | " 
    to_print *= "Box Loss: "   * bbox   * " | " 
    to_print *= "Lm Loss: "    * lm     * " | " 
    to_print *= "Decay: "      * decay  * "\n"
    return to_print
end