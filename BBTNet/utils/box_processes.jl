include("../../configs.jl")

function encode_gt_and_get_indices(gt, bboxes, pos_thold, neg_thold; dtype=Array{Float32})
    priors = _get_priorboxes(dtype=dtype)
    decoded_bboxes = _decode_bboxes(reshape(bboxes, (1, size(bboxes)...)), priors) * img_size
    decoded_bboxes = reshape(decoded_bboxes, size(decoded_bboxes)[2:end])  
    iou_vals = iou(gt[:,1:4], _to_min_max_form(bboxes), dtype=dtype) 
    
    max_gt_vals, max_gt_idx = findmax(iou_vals; dims=2) # gets max values and indices for each gt
    max_prior_vals, max_prior_idx = findmax(iou_vals; dims=1) # gets max values and indices for each prior
    
    pos_indices = getindex.(max_gt_idx[findall(max_gt_vals .>= pos_thold)], [1 2])[:, 2]
    num_pos_boxes = length(pos_indices)
    if num_pos_boxes == 0 
        # if no positive anchor boxes are found, then no loss will be calculated
        return nothing, nothing, nothing
    end 
    
    neg_indices = max_prior_idx[findall(max_prior_vals .<= neg_thold)]
    if size(neg_indices)[1] > (ohem_ratio * num_pos_boxes)
        # select the most negative ohem_ratio * num_pos_boxes many boxes
        neg_indices = getindex.(max_prior_idx[sortperm(vec(max_prior_vals))[1:ohem_ratio * num_pos_boxes]], [1 2])[:, 1]
    else
        neg_indices = getindex.(neg_indices, [1 2])[:, 1]
    end
     
    """
    Below, encoding of the initial ground truth labels will be made. In the paper the process is summarized as:
    Box_lengths     = log(gt_w / scale) + log(gt_h / scale)
    Point_coords    = [(bbox_x1 - anchor_center_x) / scale] + [(bbox_x2 - anchor_center_x) / scale] # bbox x-coord loss
                    + [(bbox_y1 - anchor_center_y) / scale] + [(bbox_y2 - anchor_center_y) / scale] # bbox y-coord loss
                    + for each landmark {[(landmark_x - anchor_center_x) / scale] + [(landmark_y - anchor_center_y) / scale]}
    
    Here, the scale is actually the width and height of the prior anchor box.
    """
  
    selected_priors = priors[pos_indices,:]
    # gt bbox conversion
    gt ./= img_size
    gt[:,1:4] = _to_center_length_form(gt[:,1:4])
    gt[:,3:4] = log.(gt[:,3:4] ./ selected_priors[:, 3:4])
    gt[:,1:2] = (gt[:,1:2] .- selected_priors[:, 1:2]) ./ selected_priors[:, 3:4]
    gt[:,5:6] = (gt[:,5:6] .- selected_priors[:, 1:2]) ./ selected_priors[:, 3:4]
    gt[:,7:8] = (gt[:,7:8] .- selected_priors[:, 1:2]) ./ selected_priors[:, 3:4]
    gt[:,9:10] = (gt[:,9:10] .- selected_priors[:, 1:2]) ./ selected_priors[:, 3:4]
    gt[:,11:12] = (gt[:,11:12] .- selected_priors[:, 1:2]) ./ selected_priors[:, 3:4]
    gt[:,13:14] = (gt[:,13:14] .- selected_priors[:, 1:2]) ./ selected_priors[:, 3:4]

    return gt, pos_indices, neg_indices
end

"""
The parts below are mostly adapted from:
* https://github.com/biubug6/Pytorch_Retinaface
* https://github.com/Hakuyume/chainer-ssd
"""

"""
!!! Each individual box in boxes has the format (x_min, y_min, x_max, y_max) !!!
Returns intersection of unions for combination of each boxes in both parameters.
A   : number of boxes in boxes1
B   : number of boxes in boxes2

Return: iou values with shape (A, B)
"""
function iou(boxes1, boxes2; dtype=Array{Float32})
    A, C = size(boxes1); B, D = size(boxes2)
    
    # if run_gpu
    #     # KnetArray does not support repeat method, thus, CuArray used here.
    #     boxes1 = CuArray(boxes1)
    #     boxes2 = CuArray(boxes2)
    # end
    
    intersections = _get_intersections(boxes1, boxes2)
    
    area1 = (boxes1[:,4] .- boxes1[:,2]) .* (boxes1[:,3] .- boxes1[:,1])
    area2 = (boxes2[:,4] .- boxes2[:,2]) .* (boxes2[:,3] .- boxes2[:,1])

    area1 = repeat(reshape(area1, (A, 1)), outer=(1, B))
    area2 = repeat(reshape(area2, (1, B)), outer=(A, 1))
    
    # if run_gpu
    #     boxes1 = convert(dtype, boxes1)
    #     boxes2 = convert(dtype, boxes2)
    #     intersections = convert(dtype, intersections)
    #     area1 = convert(dtype, area1)
    #     area2 = convert(dtype, area2)
    # end
    
    unions = area1 .+ area2 .- intersections
    return intersections ./ unions
end

"""
Conversion from: 
(center_x, center_y, width, height) --> (min_x, min_y, max_x, max_y)
If the start of a box is negative or bigger than the image size, then 
it is set to either 0 or image size.
"""
function _to_min_max_form(boxes)
    half_lens = boxes[:,3:4] ./ 2
    return cat(
        max.(boxes[:,1:2] .- half_lens, 0), 
        min.(boxes[:,1:2] .+ half_lens, img_size), dims=2)
end

"""
Conversion from: 
(min_x, min_y, max_x, max_y) --> (center_x, center_y, width, height)
"""
function _to_center_length_form(boxes)
    lengths = boxes[:,3:4] .- boxes[:,1:2]
    return cat(boxes[:,1:2] .+ (lengths ./ 2), lengths, dims=2)
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
    A, C = size(boxes1)
    B, D = size(boxes2)
    
    b1_mins = repeat(reshape(boxes1[:, 1:2], (A, 1, 2)), outer=(1, B, 1))
    b2_mins = repeat(reshape(boxes2[:, 1:2], (1, B, 2)), outer=(A, 1, 1))
    min_coords = min.(b1_mins, b2_mins)
    
    b1_maxs = repeat(reshape(boxes1[:, 3:4], (A, 1, 2)), outer=(1, B, 1))
    b2_maxs = repeat(reshape(boxes2[:, 3:4], (1, B, 2)), outer=(A, 1, 1))
    max_coords = max.(b1_maxs, b2_maxs)
    
    intersections = max.(max_coords .- min_coords, 0)
    return intersections[:,:,1] .* intersections[:,:,2]
end

"""
Returns the anchor boxes with their center_x, center_y, width, height information.
"""
function _get_priorboxes(;dtype=Array{Float64})
    feature_maps = [Int(ceil(img_size / scale["stride"])) for scale in anchor_info]
    num_proposals = num_anchors * sum([i*i for i in feature_maps])
    anchors = convert(dtype, zeros(num_proposals, 4))

    counter = 1
    for (idx, f) in enumerate(feature_maps)
        scaler = anchor_info[idx]["stride"] / img_size
        for s in anchor_info[idx]["anchors"]
            bbox_len = s / img_size
            for h in 1:f
                for w in 1:f
                    cx = (w - 0.5) * scaler
                    cy = (h - 0.5) * scaler
                    anchors[counter,:] = convert(dtype, [cx cy bbox_len bbox_len])
                    counter += 1
                end
            end
        end
    end
    return anchors
end

"""
Decoder functions:
* The main motivation is that the network does not directly predict the proposals but
the combination of prediction and prior anchor box style and rescaling gives the actual
bounding box and landmark coordinations.
"""
function decode_points(bboxes, landmarks; dtype=Array{Float64})
    priors = _get_priorboxes(dtype=dtype)
    decoded_bboxes = _decode_bboxes(bboxes, priors)
    decoded_bboxes .*= img_size
    decoded_landmarks = _decode_landmarks(landmarks, priors)
    decoded_landmarks .*= img_size
    return decoded_bboxes, decoded_landmarks
end

function _decode_bboxes(bbox, priors)
    P = size(priors)[1]
    centers = reshape(priors[:, 1:2], (1, P, 2)) + bbox[:, :, 1:2] .* reshape(priors[:, 3:end], (1, P, 2))
    lengths = exp.(reshape(priors[:, 3:end], (1, P, 2)) .* bbox[:, :, 3:end])
    centers .-= centers ./ 2
    return cat(centers, lengths, dims=3)
end

function _decode_landmarks(landmarks, priors)
    P = size(priors)[1]
    lm1 = reshape(priors[:, 1:2], (1, P, 2)) + landmarks[:, :, 1:2] .* reshape(priors[:, 3:end], (1, P, 2))
    lm2 = reshape(priors[:, 1:2], (1, P, 2)) + landmarks[:, :, 3:4] .* reshape(priors[:, 3:end], (1, P, 2))
    lm3 = reshape(priors[:, 1:2], (1, P, 2)) + landmarks[:, :, 5:6] .* reshape(priors[:, 3:end], (1, P, 2))
    lm4 = reshape(priors[:, 1:2], (1, P, 2)) + landmarks[:, :, 7:8] .* reshape(priors[:, 3:end], (1, P, 2))
    lm5 = reshape(priors[:, 1:2], (1, P, 2)) + landmarks[:, :, 9:10] .* reshape(priors[:, 3:end], (1, P, 2))
    return cat(lm1, lm2, lm3, lm4, lm5, dims=3)
end