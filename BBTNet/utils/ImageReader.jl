using Random
using Images, ImageView

include("../../configs.jl")
include("augment.jl")

mutable struct Image_Reader
    augment::Bool
    pre_scales::Array{Float64}
    function Image_Reader(augment::Bool=false)
        return new(augment, crop_ratios)
    end
end

function read_img(r::Image_Reader, dir, bboxes, len)
    img = channelview(load(dir))
    new_boxes = deepcopy(bboxes)
    if r.augment 
        # cropping the picture with a scale
        roi = _find_roi(r, img)
        img, new_boxes = crop_image(img, new_boxes, roi)
        # resizing to <len>
        img, new_boxes = resize_square_img(img, new_boxes, len, (roi[3] - roi[1]))
        # horizontal flip
        if rand() < 0.5 img, new_boxes = flip_horizontal(img, new_boxes) end
        # color distortion
        img = distort_color(img)
    else
        img, new_boxes, maxlen = squaritize_img(img, new_boxes)
        img, new_boxes = resize_square_img(img, new_boxes, len, maxlen) 
    end 
    new_boxes[new_boxes .< 0] .= -1
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