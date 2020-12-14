using Random
using Images

include("../../configs.jl")

mutable struct Image_Reader
    augment::Bool
    pre_scales::Array{Float64}
    function Image_Reader(augment::Bool=false)
        return new(augment, crop_ratios)
    end
end

function read_img(r::Image_Reader, dir, new_boxes, len)
    img = channelview(load(dir))
    if r.augment 
        roi = _find_roi(r, img)
        # horizontal flip
        if rand() < 0.5 img, new_boxes = flip_horizontal(img, new_boxes) end
        # color distortion
        img = distort_color(img)
        # cropping the picture with a scale
        img, new_boxes = crop_image(img, new_boxes, roi)
        # resizing to <len>
        img, new_boxes = resize_square_img(img, new_boxes, len, (roi[3] - roi[1]))
        
    else
        img, new_boxes, maxlen = squaritize_img(img, new_boxes)
        img, new_boxes = resize_square_img(img, new_boxes, len, maxlen) 
        new_boxes[new_boxes .< 0] .= -1
    end 
            
    bbox_indices = getindex.(findall(new_boxes[1,:] .>= 0)) # boxes having bboxes
    lm_indices = getindex.(findall(new_boxes[15,:] .>= 0)) # boxes having landmarks
    
    common_indices = []
    diff_indices = []
    for val in bbox_indices
        if val in lm_indices
            push!(common_indices, val)
        else
            push!(diff_indices, val)
        end
    end

    print("Common sizes: ", size(common_indices, 1), '\n')
    
    new_boxes = new_boxes[:,common_indices]  
    return img, new_boxes
end

function _find_roi(r::Image_Reader, img)
    c, h, w = size(img)
    maxlen = max(h, w); minlen = min(h, w)
    scale = r.pre_scales[rand(1:5)]
    new_short = Int(ceil(minlen * scale))
    x1 = 1; y1 = 1
    if scale < 1
        x1 = rand(1:max(1, w-new_short))
        y1 = rand(1:max(1, h-new_short))
    end
    x2 = (new_short-1) + x1
    y2 = (new_short-1) + y1
    #region of interest is determined
    return [x1 y1 x2 y2] 
end

function crop_image(img, bboxes, roi)
    img = img[:, roi[2]:roi[4], roi[1]:roi[3]]
    new_boxes = deepcopy(bboxes)
    box_size = size(new_boxes)
    # setting landmarks outside of the region to -1
    for person in 1:box_size[2]
        for lm in 5:2:box_size[1]-2
            # if not in the cropped area
            if (new_boxes[lm, person] > roi[3]) || 
                (new_boxes[lm, person] < roi[1]) || 
                (new_boxes[lm + 1, person] > roi[4]) || 
                (new_boxes[lm + 1, person] < roi[2])
                new_boxes[lm, person] = -1
                new_boxes[lm + 1, person] = -1
            # if in the cropped area
            else
                new_boxes[lm, person] -= roi[1]
                new_boxes[lm + 1, person] -= roi[2]
            end
        end
        if !all(>=(0), new_boxes[5:14,person])
            new_boxes[15, person] = -1.0
        end
        # setting bounding box coordinates
        
        # completely off the cropped area
        if (new_boxes[3,person] <= roi[1]) || (new_boxes[1,person] >= roi[3]) 
            new_boxes[1:4, person] .= -1
        elseif (new_boxes[4,person] <= roi[2]) || (new_boxes[2,person] >= roi[4])
            new_boxes[1:4, person] .= -1
        else
            # partial fits
            if new_boxes[3,person] > roi[3]
                new_boxes[3,person] = roi[3]
            end 
            if new_boxes[4,person] > roi[4]
                new_boxes[4,person] = roi[4]
            end  
            
            if 0 <= new_boxes[1,person] < roi[1]
                new_boxes[1,person] = 0
            else
                new_boxes[1,person] -= roi[1]
            end 
            new_boxes[3,person] -= roi[1]
            
            if 0 <= new_boxes[2,person] < roi[2]
                new_boxes[2,person] = 0
            else
                new_boxes[2,person] -= roi[2]
            end
            new_boxes[4,person] -= roi[2]
        end
    end
    
    return img, new_boxes
end

function resize_square_img(img, bboxes, new_len, old_len)
    img = imresize(img, (3, new_len, new_len))
    ratio = new_len / old_len
    bboxes .*= ratio
    bboxes[bboxes .< 0] .= -1
    return img, bboxes
end

function squaritize_img(img, bboxes)
    c, h, w = size(img)
    maxlen = max(h, w); minlen = min(h, w)
    if h == w return img, bboxes, maxlen
    else
        full_img = zeros(3, maxlen, maxlen)
        diff = maxlen - minlen
        pads = [Int(floor(diff/2)), Int(floor(diff/2))]
        # complete the length for an odd difference size
        if mod(diff, 2) == 1 pads[1] += 1 end
        if minlen == w
            full_img[:,:,pads[1]+1:maxlen-pads[2]] = img
            bboxes[1:2:13,:] .+= pads[1]
        else
            full_img[:,pads[1]+1:maxlen-pads[2],:] = img
            bboxes[2:2:14,:] .+= pads[1]
        end
        return full_img, bboxes, maxlen
    end
end

function flip_horizontal(img, bboxes)
    img = reverse(img, dims=3)
    bboxes[1:2:13,:] .= size(img, 3) .- bboxes[1:2:13,:]
    temp = bboxes[1,:]
    bboxes[3,:] .= bboxes[1,:]
    bboxes[3,:] .= temp
    bboxes[bboxes .> size(img, 3)] .= -1
    return img, bboxes
end

function distort_color(img)
    # brightness distortion
    if rand() < 0.5 img = _convert(img, beta=rand(-0.125:0.125)) end 
    # contrast distortion
    if rand() < 0.5 img = _convert(img, alpha=rand(0.5:1.5)) end
    
    img = channelview(colorview(HSV, float.(img)))
    # saturation distortion
    if rand() < 0.5 img[2,:,:] = _convert(img[2,:,:], alpha=rand(0.5:1.5)) end
    # hue distortion
    if rand() < 0.5 img[1,:,:] = _convert(img[1,:,:], beta=rand(-(18/256):(18/256))) end
    
    img = channelview(colorview(RGB, colorview(HSV, img)))
    return img
end

function _convert(image; alpha=1, beta=0)
    image = image .* alpha .+ beta
    image[image .< 0] .= 0
    image[image .> 1] .= 1
    return image
end