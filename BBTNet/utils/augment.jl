using Images, ImageView

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
        if all(x->x < 0, new_boxes[5:14, person])
            new_boxes[15, person] = -1.0
        end
        # setting bounding box coordinates
        
        # completely off the cropped area
        if (new_boxes[1,person] + new_boxes[3,person] <= roi[1]) || (new_boxes[1,person] >= roi[3]) 
            new_boxes[1:4, person] .= -1
        elseif (new_boxes[2,person] + new_boxes[4,person] <= roi[2]) || (new_boxes[2,person] >= roi[4])
            new_boxes[1:4, person] .= -1
        else
            # partial fits
            if new_boxes[1,person] + new_boxes[3,person] > roi[3]
                new_boxes[3,person] = roi[3] - new_boxes[1,person]
            end 
            if new_boxes[2,person] + new_boxes[4,person] > roi[4]
                new_boxes[4,person] = roi[4] - new_boxes[2,person]
            end        
            if 0 <= new_boxes[1,person] < roi[1]
                new_boxes[1,person] = 0
                new_boxes[3,person] -= roi[1] - new_boxes[1,person]
            else
                new_boxes[1,person] -= roi[1]
            end     
            if 0 <= new_boxes[2,person] < roi[2]
                new_boxes[2,person] = 0
                new_boxes[4,person] -= roi[2] - new_boxes[2,person]
            else
                new_boxes[2,person] -= roi[2]
            end    
        end
    end
    return img, new_boxes
end

function resize_square_img(img, bboxes, new_len, old_len)
    img = imresize(img, (3, new_len, new_len))
    ratio = new_len / old_len
    bboxes .*= ratio
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
            full_img[:,pads[1]+1:maxlen-pads[2]] = img
            bboxes[1:2:13,:] .+= pads[1]
            bboxes[1,:] .+= pads[1]
        else
            full_img[:,pads[1]+1:maxlen-pads[2],:] = img
            bboxes[6:2:14,:] .+= pads[1]
            bboxes[2,:] .+= pads[1]
        end
        return full_img, bboxes, maxlen
    end
end

function flip_horizontal(img, bboxes)
    img = reverse(img, dims=2)
    bboxes[6:2:14,:] .= size(img)[2] .- bboxes[6:2:14,:]
    bboxes[2,:] .= (size(img)[2] .- bboxes[2,:]) .- bboxes[4,:]
    bboxes[bboxes .> size(img)[2]] .= -1
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