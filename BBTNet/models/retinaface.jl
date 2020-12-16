using ProgressBars, Printf

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
        push!(layers, Conv2D(1, 1, input_dim, num_anchors*task_len, dtype=dtype, bias=false))
    end
    return HeadGetter(layers, task_len)
end

function (hg::HeadGetter)(xs; train=true)
    proposals = []
    for (i, x) in enumerate(xs)
        proposal = hg.layers[i](x, train=train)
        batch_size = size(proposal)[end]
        num_proposals = Int(floor(prod(size(proposal)) / (hg.task_len * batch_size)))
        proposal = reshape(proposal, (batch_size, num_proposals, hg.task_len))
        push!(proposals, proposal)
    end
    if hg.task_len == 2 softmax(cat(proposals..., dims=2), dims=3)
    else return cat(proposals..., dims=2)
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

function RetinaFace(;dtype=Array{Float64}) 
    
    return RetinaFace(
        ResNet50(include_top=false, dtype=dtype), 
        FPN(dtype=dtype), SSH(dtype=dtype), SSH(dtype=dtype),
        HeadGetter(256, num_anchors, 2, dtype=dtype), HeadGetter(256, num_anchors, 2, dtype=dtype),
        HeadGetter(256, num_anchors, 4, dtype=dtype), HeadGetter(256, num_anchors, 4, dtype=dtype),
        HeadGetter(256, num_anchors, 10, dtype=dtype), HeadGetter(256, num_anchors, 10, dtype=dtype),
        dtype
    )   
end

# mode 1 means first context head, 2 means second context head, 0 means no context head
function (model::RetinaFace)(x, y=nothing, mode=0, train=true, weight_decay=0)
    # first processes
    c2, c3, c4, c5 = model.backbone(x, return_intermediate=true, train=train)
    p_vals = model.fpn([c2, c3, c4, c5], train=train)
    # print("Passed backbone and FPN structures.\n")  
    
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
    # print("Passed Context Head structures.\n")

    if y === nothing && train == false
        # returning direct values for evaluation
        return class_vals, bbox_vals, landmark_vals 
    elseif y === nothing
        # for predicting, the founded boxes should be decoded to their real values
        bboxes_decoded, landmarks_decoded = decode_points(bbox_vals, landmark_vals)
        print("Predicted points are decoded.\n")
        
        indices = findall(class_vals[:,:,1] .>= conf_level)
        N = size(class_vals)[1]
        boxes = []
        for n in 1:N push!(boxes, []) end
        for idx in indices push!(boxes[idx[1]], idx[2]) end

        cl_results = []; bbox_results = []; landmark_results = []
        for n in 1:N
            push!(cl_results, class_vals[n, boxes[n], :])
            push!(bbox_results, bboxes_decoded[n, boxes[n], :])
            push!(landmark_results, landmarks_decoded[n, boxes[n], :])
        end

        print("Returning prediction results above confidence level: ", conf_level, ".\n")
        return  cl_results, bbox_results, landmark_results 
    else
        # for training, loss will be calculated and returned
        affected_loss = 0
        loss_val = 0
        pos_thold = mode == 1 ? head1_pos_iou : head2_pos_iou
        neg_thold = mode == 1 ? head1_neg_iou : head2_neg_iou
        
        N = size(class_vals, 1)
        batch_gt = cat(value(bbox_vals), value(landmark_vals), dims=3)
        batch_cls = convert(Array{Int64}, ones(N, size(class_vals, 2)))
        
        for n in 1:size(class_vals, 1)
            # loop for each input in batch, since all inputs may have different number of boxes
            if isempty(y[n]) || (y[n] == []) || (y[n] === nothing)
                continue 
            end 
            
            bboxes = Array(value(bbox_vals[n,:,:]))
            gt, pos_indices, neg_indices = encode_gt_and_get_indices(permutedims(y[n],(2, 1)), bboxes, pos_thold, neg_thold)
            if pos_indices === nothing 
                continue 
            end 
            
            gt = convert(model.dtype, gt)
            batch_gt[n,pos_indices,:] = gt[:,1:14] 
            batch_cls[n, pos_indices] .= 1
            batch_cls[n, neg_indices] .= 2
            affected_loss += 1
        end
        if affected_loss > 0 
            class_vals = reshape(class_vals, (size(class_vals, 3), prod(size(class_vals)[1:2])))
            batch_cls = reshape(batch_cls, (1, prod(size(batch_cls))))
            loss_val += nll(class_vals, batch_cls)
#             print("After nll: ", loss_val, "\n")
        end
        loss_val += lambda1 * smooth_l1(abs.(batch_gt[:,:,1:4] .- bbox_vals))
#         print("After bbox: ", loss_val, "\n")
        loss_val += lambda2 * smooth_l1(abs.(batch_gt[:,:,5:end] .- landmark_vals))
#         print("After lm: ", loss_val, "\n")
        
        if loss_val > 0
            loss_val /= affected_loss
            if weight_decay > 0
                for p in params(model)
                    # weight decay process
                    loss_val -= weight_decay * sum(p.^2)
                end
            end
        end
        # print("\nFinal loss: ", loss_val, '\n')
        return loss_val
    end
end

function train_model(model::RetinaFace, data_reader; val_data=nothing)
    print("\n============================== TRAINING PROCESS ==============================\n\n")
    loss_history = []

    for e in 1:num_epochs
        (imgs, boxes), state = iterate(data_reader)
        iter_no = 1
        last_loss = 0
        total_batches = size(state, 1) + size(imgs)[end]
        curr_batch = ProgressBar(1:total_batches, width =100)
        
        while state !== nothing 
            set_description(curr_batch, string(@sprintf("Epoch: %d", e)))
            set_postfix(curr_batch, Loss=@sprintf("%.2f", last_loss))
            if e < lr_change_epoch[1]
                momentum!(model, [(imgs, boxes, mode, true, weight_decay)], lr=lrs[1], gamma=momentum)
            elseif e < lr_change_epoch[2]
                momentum!(model, [(imgs, boxes, mode, true, weight_decay)], lr=lrs[2], gamma=momentum)
            elseif e < lr_change_epoch[3]
                momentum!(model, [(imgs, boxes, mode, true, weight_decay)], lr=lrs[3], gamma=momentum)
            else
                momentum!(model, [(imgs, boxes, mode, true, weight_decay)], lr=lrs[4], gamma=momentum)
            end
            if mod(iter_no, 20) == 0 
                last_loss = model(imgs, boxes, train=false)
            end
            (imgs, boxes), state = iterate(data_reader, state)
            for _ in 1:size(imgs)[end] iterate(curr_batch, 1) end
            iter_no += 1
        end
        
        # Evaluate both training and val data after each epoch.
        train_loss = evaluate_model(model, data_reader)
        print("\nEpoch: ", e, " ---> Train Loss: ", train_loss)
        if val_data !== nothing
            val_loss = evaluate_model(model, val_data)
            push!(loss_history, (train_loss, val_loss))
            print(" || Validation Loss: ", val_loss)
        else
            push!(loss_history, train_loss)
        end
        print("\n")
    end
    return loss_history
end

function evaluate_model(model::RetinaFace, data_reader)
    (imgs, boxes), state = iterate(data_reader)
    imgs = convert(model.dtype, permutedims(imgs, (3,2,1,4)))
    boxes = convert(model.dtype, boxes)
    num_iters = 0
    loss_val = 0.0
    while state !== nothing
        loss_val += model(imgs, y=boxes, train=false)
        num_iters += 1
        (imgs, boxes), state = iterate(data_reader)
        imgs = convert(model.dtype, permutedims(imgs, (3,2,1,4)))
        boxes = convert(model.dtype, boxes)
    end
    return loss_val / num_iters
end