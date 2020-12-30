include("../../configs.jl")

function encode_gt_and_get_indices(gt, priors, losses, pos_thold, neg_thold)
    iou_vals = iou(gt[1:4,:], _to_min_max_form(priors))
    # gets max values and indices for each gt
    max_gt_vals, max_gt_idx = findmax(iou_vals; dims=1)
    # gets max values and indices for each prior
    max_prior_vals, max_prior_idx = findmax(iou_vals; dims=2) 
    # selecting positive prior boxes  
    pos_selected = findall(max_gt_vals .>= pos_thold)
    pos_gt_indices = getindex.(pos_selected, [1 2])[:, 2]
    pos_prior_indices = getindex.(max_gt_idx[pos_selected], [1 2])[:, 1] 
    
    num_poses = length(pos_gt_indices)
    if num_poses == 0 
        # if no positive anchor boxes are found, then no loss will be calculated
        return nothing, nothing, nothing
    end
    
    #selecting negative prior boxes
    neg_indices = getindex.(findall(max_prior_vals .<= neg_thold), [1 2])[:,1]
    neg_indices = neg_indices[sortperm(losses[neg_indices], rev=true)]
    neg_indices = neg_indices[1:ohem_ratio * num_poses]
            
    # gt bbox conversion
    selected_priors = priors[:,pos_prior_indices]
    gt = gt[:,pos_gt_indices] # only positive ground truth values are included
    
    gt[1:4,:] = _to_center_length_form(gt[1:4,:])
    gt[3:4,:] = log.(gt[3:4,:] ./ (variances[1] .* selected_priors[3:4,:]))
    
    gt[1:2,:] = (gt[1:2,:] .- selected_priors[1:2,:]) ./ (variances[2] .* selected_priors[3:4,:])
    gt[5:6,:] = (gt[5:6,:] .- selected_priors[1:2,:]) ./ (variances[2] .* selected_priors[3:4,:])
    gt[7:8,:] = (gt[7:8,:] .- selected_priors[1:2,:]) ./ (variances[2] .* selected_priors[3:4,:])
    gt[9:10,:] = (gt[9:10,:] .- selected_priors[1:2,:]) ./ (variances[2] .* selected_priors[3:4,:])
    gt[11:12,:] = (gt[11:12,:] .- selected_priors[1:2,:]) ./ (variances[2] .* selected_priors[3:4,:])
    gt[13:14,:] = (gt[13:14,:] .- selected_priors[1:2,:]) ./ (variances[2] .* selected_priors[3:4,:])
    
    return gt, pos_prior_indices, neg_indices
end



function nms(scores, points)
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
            if val[1] <= nms_threshold
                push!(keep, i)
            end
        end   
    end
    return keep
end

"""
!!! Each individual box in boxes has the format (x_min, y_min, x_max, y_max) !!!
Returns intersection of unions for combination of each boxes in both parameters.
A   : number of boxes in boxes1
B   : number of boxes in boxes2

Return: iou values with shape (A, B)
"""
function iou(boxes1, boxes2)
    C, A = size(boxes1); D, B = size(boxes2);
    area1 = (boxes1[4,:] .- boxes1[2,:]) .* (boxes1[3,:] .- boxes1[1,:])
    area2 = (boxes2[4,:] .- boxes2[2,:]) .* (boxes2[3,:] .- boxes2[1,:]) 
    intersections = _get_intersections(boxes1, boxes2) 
    unions = reshape(area1, (1, A)) .+ reshape(area2, (B, 1)) .- intersections
    return intersections ./ unions
end

"""
!!! Each individual box in boxes has the format (x_min, y_min, x_max, y_max) !!!
We resize both boxes to [A,B,4]:
* [A,2] -> [A,1,2] -> [A,B,2]
* [B,2] -> [1,B,2] -> [A,B,2]
Then we compute the area of intersect between box_a and box_b.

Return: intersection area with shape (A, B)
"""
function _get_intersections(boxes1, boxes2)
    C, A = size(boxes1); D, B = size(boxes2);
    min_coords = max.(reshape(boxes1[1:2,:], (2, 1, A)), reshape(boxes2[1:2,:], (2, B, 1)))
    max_coords = min.(reshape(boxes1[3:4,:], (2, 1, A)), reshape(boxes2[3:4,:], (2, B, 1)))  
    intersections = max.(max_coords .- min_coords, 0)
    return intersections[1,:,:] .* intersections[2,:,:]
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
    # converting to 2D
    boxes = reshape(boxes, (size(boxes)[1], prod(size(boxes)[2:end]))) # converting to 2D
    
    lengths = boxes[3:4,:] .- boxes[1:2,:]
    return reshape(cat(boxes[1:2,:] .+ (lengths ./ 2), lengths, dims=1), init_shape)
end


"""
Returns the anchor boxes with their center_x, center_y, width, height information.
"""
function _get_priorboxes()
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
    if length(size(priors)) == 2
        priors = reshape(priors, (size(priors)..., 1))
    end
    centers = priors[1:2,:,:] .+ bbox[1:2,:,:] .* variances[2] .* priors[3:end,:,:]
    lengths = priors[3:end,:,:] .* exp.(bbox[3:end,:,:] .* variances[1])
    return cat(centers, lengths, dims=1)
end

function _decode_landmarks(landmarks, priors)
    if length(size(priors)) == 2
        priors = reshape(priors, (size(priors)..., 1))
    end
    lm1 = priors[1:2,:,:] .+ landmarks[1:2,:,:]  .* variances[2] .* priors[3:end,:,:]
    lm2 = priors[1:2,:,:] .+ landmarks[3:4,:,:]  .* variances[2] .* priors[3:end,:,:]
    lm3 = priors[1:2,:,:] .+ landmarks[5:6,:,:]  .* variances[2] .* priors[3:end,:,:]
    lm4 = priors[1:2,:,:] .+ landmarks[7:8,:,:]  .* variances[2] .* priors[3:end,:,:]
    lm5 = priors[1:2,:,:] .+ landmarks[9:10,:,:] .* variances[2] .* priors[3:end,:,:]
    return cat(lm1, lm2, lm3, lm4, lm5, dims=1)
end