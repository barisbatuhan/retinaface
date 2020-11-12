using ImageDraw

function draw_boxes_and_landmarks(img, boxes)
    to_draw = deepcopy(RGB{N0f8}.(img[1,:,:], img[2,:,:], img[3,:,:]))
    colors = [
        RGB{N0f8}(1.0,0.0,0.0), # red
        RGB{N0f8}(0.0,0.0,1.0), # blue
        RGB{N0f8}(0.0,1.0,0.0), # green
        RGB{N0f8}(1.0,1.0,0.0), # yellow
        RGB{N0f8}(0.0,1.0,1.0), # turquoise
        RGB{N0f8}(1.0,0.0,1.0) # magenta
    ]
    for person in 1:size(boxes)[2]
        rect = Int.(ceil.(boxes[:, person]))
        if any(x->x >= 0, boxes[1:4, person])
            to_draw = draw!(to_draw, Polygon([
                        (rect[1, 1], rect[2, 1]),
                        (rect[1, 1] + rect[3, 1], rect[2, 1]),
                        (rect[1, 1] + rect[3, 1], rect[2, 1] + rect[4, 1]),
                        (rect[1, 1], rect[2, 1] + rect[4, 1])]), colors[1])
        end
        colorid = 2
        for lm in 5:2:14
            if rect[lm,1] >= 0 && rect[lm+1,1] >= 0
               to_draw = draw!(to_draw, CirclePointRadius(rect[lm,1],rect[lm+1,1],2), colors[colorid]) 
            end
            colorid += 1
        end
        
    end
    return to_draw
end