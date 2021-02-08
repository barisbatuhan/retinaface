import Base: length, size, iterate, eltype, IteratorSize, IteratorEltype, haslength, @propagate_inbounds, repeat, rand, tail
import .Iterators: cycle, Cycle, take

include("../../DeepJulia/DeepJulia.jl")
include("../../configs.jl")

mutable struct WIDER_Data
    tr::Transforms
    train_data::Bool; img_size::Int;
    
    function WIDER_Data(dir, label_dir; batch_size::Int64=32, train::Bool=true, img_size=640, shuffle::Bool=true)
        files = []
        bboxes = []
        if label_dir === nothing
            bboxes = nothing
        else
            bbox_dict = Dict()
            open(label_dir * "label.txt", "r") do io
                lines = readlines(io)
                filename = nothing
                for line in lines
                    if line[1] == '#'
                        if filename !== nothing
                            push!(bboxes, _clean_bboxes(bbox_dict[filename]))
                        end
                        filename = dir * "images/" * line[3:end]
                        push!(files, filename)
                        bbox_dict[filename] = []
                    else
                        bbox = [parse(Float64, x) for x in split(line, " ")]
                        bbox[3] += bbox[1]
                        bbox[4] += bbox[2]
                        push!(bbox_dict[filename], bbox)
                    end
                end
                push!(bboxes, _clean_bboxes(bbox_dict[filename]))
            end
        end

        processes = [];

        if train
            push!(processes, DistortColor()) 
            push!(processes, RandomCrop(min_ratio=0.5))
            push!(processes, Flip())
            push!(processes, Resize(img_size, img_size))
        else
            img_size = -1
            push!(processes, Squaritize(fill_value=0.5))
        end
        
        tr = Transforms(
            processes, convert(Array{String}, files), labels=bboxes, batch_size=batch_size, 
            img_size=img_size, shuffle=shuffle, return_changes=true)
        
        return new(tr, train, img_size)
    end
end

function _clean_bboxes(bboxes)
    annotations = zeros(15, length(bboxes))
    # annotation processing
    for person in 1:length(bboxes)
        z_coords = 0
        iter_cnt = length(bboxes[person]) == 4 ? 4 : length(bboxes[person])-1 # val data and train data differences
        for iter in 1:iter_cnt
            if (iter > 4) && (mod(iter, 3) == 1)
                z_coords += 1
                continue
            end
            i = iter - z_coords
            annotations[i,person] =  bboxes[person][iter]
        end
        if all(>=(0), annotations[5:14,person]) 
            annotations[15,person] = 1.0
        else # no/partial landmark info
            annotations[15,person] = -1.0
        end
        if !all(>=(0), annotations[1:4,person]) 
            annotations[1,person] = -1.0
        end
    end
    return annotations
end

function iterate(data::WIDER_Data; restart::Bool=false)
    
    imgs, labels, changes = get_batch(data.tr, restart=restart)   
    if imgs === nothing
        return nothing, nothing
    
    elseif labels !== nothing
        # for each image in a batch, labels should be corrected.
        for n in 1:size(imgs)[end] 
            change = changes[n]
            
            if data.train_data
                # random crop
                crop_ch = change["crop"]; len = crop_ch[3] - crop_ch[1];
                labels[n][1:2:14,:] .-= crop_ch[1]
                labels[n][2:2:14,:] .-= crop_ch[2]
                ## fixing the left side to 0s if they are out of the range
                outrangex = findall(labels[n][[1,2],:] .< 0)
                labels[n][outrangex] .= 0
                ## right side of the box is out of the range
                rightrangex = getindex.(findall(labels[n][3,:] .>= len))
                labels[n][3, rightrangex] .= len
                rightrangey = getindex.(findall(labels[n][4,:] .>= len))
                labels[n][4, rightrangey] .= len
                ## if box is completely out of the range
                left_search = findall(labels[n][[1, 2],:] .>= len)
                if length(left_search) > 0
                    leftout = getindex.(left_search, [1 2])[:, 2]
                    labels[n][:,leftout] .= -1
                end
                right_search = findall(labels[n][[3, 4],:] .<= 0)
                if length(right_search) > 0
                    rightout = getindex.(right_search, [1 2])[:, 2]
                    labels[n][:,rightout] .= -1
                end
                ## if a landmark is out of the range
                right_search = findall(labels[n][5:1:14,:] .>= len)
                if length(right_search) > 0
                    rightrange = getindex.(right_search, [1 2])[:, 2]
                    labels[n][5:15, rightrange] .= -1
                end
                left_search = findall(labels[n][5:1:14,:] .< 0)
                if length(left_search) > 0
                    leftrange = getindex.(left_search, [1 2])[:, 2]
                    labels[n][5:15, leftrange] .= -1
                end
                # flip
                crop_ch = change["flip"]
                if crop_ch[1]
                    labels[n][1:2:14,:] .= len .- labels[n][1:2:14,:]
                    temp = labels[n][3,:]
                    labels[n][3,:] .= labels[n][1,:]
                    labels[n][1,:] .= temp
                    labels[n][labels[n] .> len] .= -1
                end
                # resizing
                res_ch = change["resize"];
                labels[n][1:2:14,:] ./= res_ch[1]
                labels[n][2:2:14,:] ./= res_ch[2]
                labels[n][labels[n] .< 0] .= -1
                # eliminating the ones outside of the ROI area
                bbox_indices = getindex.(findall(labels[n][1,:] .>= 0))
                labels[n] = labels[n][:,bbox_indices]  
            else
                sq_ch = change["squaritize"];
                # squaritizing 
                labels[n][1:2:14,:] .+= sq_ch[1]
                labels[n][2:2:14,:] .+= sq_ch[2]
                # resizing 
                res_ch = change["resize"];
                labels[n][1:2:14,:] ./= res_ch[1]
                labels[n][2:2:14,:] ./= res_ch[2]
            end
        end
    end

    return imgs, labels
end