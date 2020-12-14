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
function (model::RetinaFace)(x, y=nothing; mode=0, train=true)
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
        
        for n in 1:size(class_vals, 1)
            # loop for each input in batch, since all inputs may have different number of boxes
            if isempty(y[n]) continue end
            
            bboxes = Array(value(bbox_vals))[n,1:end,1:end]
            gt, pos_indices, neg_indices = encode_gt_and_get_indices(permutedims(y[n],(2, 1)), bboxes, pos_thold, neg_thold)
            if (gt === nothing || isempty(gt)) continue end 
            gt = convert(model.dtype, gt)
            
            # Positive Losses
            loss_val += smooth_l1(gt[:,1:4], bbox_vals[n,pos_indices,1:4])  # bounding box loss
            loss_val += smooth_l1(gt[:,5:14], landmark_vals[n,pos_indices,:]) # landmark loss           
            loss_val += nll(permutedims(class_vals[n,pos_indices,:], (2, 1)), fill(1, (length(pos_indices),)))
            # Negative Losses
            loss_val += nll(permutedims(class_vals[n,neg_indices,:], (2, 1)), fill(2, (length(neg_indices),)))
            affected_loss += 1
        end
        if loss_val > 0
            loss_val /= affected_loss
            print("Loss Calculated: ", loss_val, "\n")
        end
        return loss_val
    end
end

function train_model(model::RetinaFace, data_reader; val_data=nothing)
    lr = [1e-3, 1e-2, 1e-3, 1e-4]
    lr_change_epochs = [5, 55, 68]
    # momentum=0.9; weight_decay = 0.0005
    num_epochs = 80
    loss_history = []

    for e in 1:num_epochs
        iter_no = 1
        (imgs, boxes), state = iterate(data_reader)
        # imgs = convert(model.dtype, permutedims(imgs, (3,2,1,4)))
        # boxes = convert(model.dtype, boxes)
        while state !== nothing
            if e < lr_change_epochs[1]
                momentum!(model, [(imgs, boxes)], lr=lr[1], gamma=0.9)
            elseif e < lr_change_epochs[2]
                momentum!(model, [(imgs, boxes)], lr=lr[2], gamma=0.9)
            elseif e < lr_change_epochs[3]
                momentum!(model, [(imgs, boxes)], lr=lr[3], gamma=0.9)
            else
                momentum!(model, [(imgs, boxes)], lr=lr[4], gamma=0.9)
            end
            if mod(iter_no, 20) == 0 
                loss_val = model(imgs, boxes, train=false)
                print("In epoch: ", e, " & Iter: ", iter_no, "--> Loss: ", loss_val,"\n")
            end
            (imgs, boxes), state = iterate(data_reader, state)
            # imgs = convert(model.dtype, permutedims(imgs, (3,2,1,4)))
            # boxes = convert(model.dtype, boxes)
            iter_no += 1
        end
        # Evaluate both training and val data after each epoch.
        train_loss = evaluate_model(model, data_reader)
        print("Epoch: ", e, " ---> Train Loss: ", train_loss)
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