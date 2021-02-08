using Images, ImageDraw, Plots

function draw_boxes_and_landmarks(img, boxes; lms=nothing, conf=nothing, len=640)
    to_draw = deepcopy(RGB{N0f8}.(img[1,:,:], img[2,:,:], img[3,:,:]))
    colors = [
        RGB{N0f8}(1.0,0.0,0.0), # red
        RGB{N0f8}(0.0,0.0,1.0), # blue
        RGB{N0f8}(0.0,1.0,0.0), # green
        RGB{N0f8}(1.0,1.0,0.0), # yellow
        RGB{N0f8}(0.0,1.0,1.0), # turquoise
        RGB{N0f8}(1.0,0.0,1.0) # magenta
    ]
    
    for person in 1:size(boxes, 2)
        rect = Int.(ceil.(boxes[:, person]))
        if any(x->x >= 0, boxes[1:4, person])
            to_draw = draw!(to_draw, Polygon([
                        (rect[1, 1], rect[2, 1]),
                        (rect[3, 1], rect[2, 1]),
                        (rect[3, 1], rect[4, 1]),
                        (rect[1, 1], rect[4, 1])]), colors[1])
        end
        colorid = 2
        if lms !== nothing
            pts = Int.(ceil.(lms[:, person]))
            for lm in 1:2:10
                if pts[lm,1] >= 0 && pts[lm+1,1] >= 0
                   to_draw = draw!(to_draw, CirclePointRadius(pts[lm,1],pts[lm+1,1],2), colors[colorid]) 
                end
                colorid += 1
            end
        end
    end
    
    if conf !== nothing
        plt = plot(to_draw, size=(len,len))
        for person in 1:size(conf)[2]
            pt_x = boxes[1, person] + (boxes[3, person] - boxes[1, person]) / 2
            pt_y = boxes[2, person] - 10
            annotate!(pt_x, pt_y, text(string(round.(value(conf[2,person]); digits=4)), :orange, 6))
        end
        return plt
    else
        return to_draw
    end
    
end