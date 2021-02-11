
include("../../configs.jl")
include("metrics.jl")

function encode_gt_and_get_indices(gt, priors, losses, pos_thold, neg_thold)
    # variances = size(priors, 2) == 102300 ? [1, 1] : [0.2, 0.1]
    variances = [1, 1]
    iou_vals = iou(gt[1:4,:], _to_min_max_form(priors))
    # selecting positive prior boxes  
    pos_pairs = getindex.(findall(iou_vals .>= pos_thold), [1 2])
    gt_idx = pos_pairs[:, 2]; prior_idx = pos_pairs[:, 1];
    
    num_poses = length(prior_idx)
    if num_poses == 0 
        # if no positive anchor boxes are found, then no loss will be calculated
        return nothing, nothing, nothing
    end

    # enlarging gt matrix to match selected anchors
    pos_gt = zeros(15, num_poses); pos_gt .= gt[:,gt_idx];
    
    #selecting negative prior boxes
    max_prior_vals, max_prior_idx = findmax(iou_vals; dims=2) 
    neg_indices = getindex.(findall(max_prior_vals .<= neg_thold), [1 2])[:,1]
    neg_cnt = ohem_ratio * num_poses
    neg_indices = neg_indices[partialsortperm(losses[neg_indices], 1:neg_cnt)]
            
    # gt bbox conversion
    selected_priors = priors[:,prior_idx]
    pos_gt[1:4,:] = _to_center_length_form(pos_gt[1:4,:])
    
    pos_gt[3:4,:] = log.(pos_gt[3:4,:] ./ selected_priors[3:4,:]) ./ variances[1]
    
    pos_gt[1:2,:] = (pos_gt[1:2,:] .- selected_priors[1:2,:]) ./ (variances[2] .* selected_priors[3:4,:])
    pos_gt[5:6,:] = (pos_gt[5:6,:] .- selected_priors[1:2,:]) ./ (variances[2] .* selected_priors[3:4,:])
    pos_gt[7:8,:] = (pos_gt[7:8,:] .- selected_priors[1:2,:]) ./ (variances[2] .* selected_priors[3:4,:])
    pos_gt[9:10,:] = (pos_gt[9:10,:] .- selected_priors[1:2,:]) ./ (variances[2] .* selected_priors[3:4,:])
    pos_gt[11:12,:] = (pos_gt[11:12,:] .- selected_priors[1:2,:]) ./ (variances[2] .* selected_priors[3:4,:])
    pos_gt[13:14,:] = (pos_gt[13:14,:] .- selected_priors[1:2,:]) ./ (variances[2] .* selected_priors[3:4,:])
    
    return pos_gt, prior_idx, neg_indices
end

function nms(scores, points; thold=0.4)
    x1 = points[1,:]; y1 = points[2,:]; x2 = points[3,:]; y2 = points[4,:];
    
    areas = (x2 - x1) .* (y2 - y1)
    order = sortperm(scores, rev=true) 
    keep = []
    
    for i in order
        if keep == []
            push!(keep, i)
        else
            iou_vals = iou(points[:,i:i], points[:,keep])
            val, idx = findmax(iou_vals, dims=1)
            if val[1] <= thold
                push!(keep, i)
            end
        end   
    end
    return keep
end

"""
Conversion from: 
(center_x, center_y, width, height) --> (min_x, min_y, max_x, max_y)
"""
function _to_min_max_form(boxes)
    init_shape = size(boxes)
    boxes = reshape(boxes, (size(boxes)[1], prod(size(boxes)[2:end]))) # converting to 2D
    half_lens = boxes[3:4,:] ./ 2
    return reshape(cat(boxes[1:2,:] .- half_lens, boxes[1:2,:] .+ half_lens, dims=1), init_shape)
end

"""
Conversion from: 
(min_x, min_y, max_x, max_y) --> (center_x, center_y, width, height)
"""
function _to_center_length_form(boxes)
    init_shape = size(boxes)
    boxes = reshape(boxes, (size(boxes)[1], prod(size(boxes)[2:end]))) # converting to 2D
    lengths = boxes[3:4,:] .- boxes[1:2,:]
    return reshape(cat(boxes[1:2,:] .+ (lengths ./ 2), lengths, dims=1), init_shape)
end


"""
Returns the anchor boxes with their center_x, center_y, width, height information.
"""
function _get_priorboxes(num_anchors, anchor_info, img_size)
    feature_maps = [Int(ceil(img_size / scale["stride"])) for scale in anchor_info]
    num_proposals = num_anchors * sum([i*i for i in feature_maps])
    anchors = zeros(4, num_proposals)

    counter = 1
    for (idx, f) in enumerate(feature_maps)
        scaler = anchor_info[idx]["stride"]
        for h in 1:f   
            cy = (h - 0.5) * scaler
            for w in 1:f
                cx = (w - 0.5) * scaler
                for s in anchor_info[idx]["anchors"]
                    anchors[:,counter] = [cx cy s s]
                    counter += 1
                end
            end
        end
    end
    return anchors
end

"""
Decoder functions:
The main motivation is that the network does not directly predict the proposals but
the combination of prediction and prior anchor box style and rescaling gives the actual
bounding box and landmark coordinations.
"""
function decode_points(bboxes, landmarks, priors)
    if length(size(priors)) == 2
        priors = reshape(priors, (size(priors)..., 1))
    end
    decoded_bboxes = _decode_bboxes(bboxes, priors)
    decoded_landmarks = _decode_landmarks(landmarks, priors)
    return decoded_bboxes, decoded_landmarks
end


function _decode_bboxes(bbox, priors)
    # variances = size(priors, 2) > 100000 ? [1, 1] : [0.2, 0.1]
    variances = [1, 1]
    if length(size(priors)) == 2
        priors = reshape(priors, (size(priors)..., 1))
    end
    centers = priors[1:2,:,:] .+ bbox[1:2,:,:] .* variances[2] .* priors[3:4,:,:]
    lengths = exp.(bbox[3:4,:,:] .* variances[1]) .* priors[3:4,:,:]
    return cat(centers, lengths, dims=1)
end

function _decode_landmarks(landmarks, priors)
    # variances = size(priors, 2) > 100000 ? [1, 1] : [0.2, 0.1]
    variances = [1, 1]
    if length(size(priors)) == 2
        priors = reshape(priors, (size(priors)..., 1))
    end
    lm1 = priors[1:2,:,:] .+ landmarks[1:2,:,:]  .* variances[2] .* priors[3:4,:,:]
    lm2 = priors[1:2,:,:] .+ landmarks[3:4,:,:]  .* variances[2] .* priors[3:4,:,:]
    lm3 = priors[1:2,:,:] .+ landmarks[5:6,:,:]  .* variances[2] .* priors[3:4,:,:]
    lm4 = priors[1:2,:,:] .+ landmarks[7:8,:,:]  .* variances[2] .* priors[3:4,:,:]
    lm5 = priors[1:2,:,:] .+ landmarks[9:10,:,:] .* variances[2] .* priors[3:4,:,:]
    return cat(lm1, lm2, lm3, lm4, lm5, dims=1)
end