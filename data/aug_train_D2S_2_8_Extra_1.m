clearvars; clearvars -global; clc; close all; warning off;
sceneFolder = 'D:/LF-E2E/data/TrainingData/Training';
patch_size = 256;
in_patch_size = 32;
cut_border = 20;
count = 1;
list = dir([sceneFolder, '/*', []]);
dataNames = setdiff({list.name}, {'.', '..'});
dataPaths = strcat(strcat(sceneFolder, '/'), dataNames);
numDatasets = length(dataNames);for ns = 1:numScenes

    numImgsX = 14;
    numImgsY = 14;
    
    resultPath = [scenePaths{ns}];
    
%%% converting the extracted light field to a different format
    inputImg = im2double(imread(resultPath));
    inputImg = rgb2ycbcr(inputImg);
    inputImg = inputImg(:,:,1);

    h = size(inputImg, 1) / numImgsY;
    w = size(inputImg, 2) / numImgsX;
    
    fullLF = zeros(h, w, numImgsY, numImgsX);
    
    for ax = 1 : numImgsX
        for ay = 1 : numImgsY
            fullLF(:, :, ay, ax) = inputImg(ay:numImgsY:end, ax:numImgsX:end);
        end
    end
    
    fullLF = fullLF(cut_border+1:h-cut_border, cut_border+1:w-cut_border, 4:11, 4:11); % we only take the 8 middle images
    
    h = h - 2 * cut_border;
    w = w - 2 * cut_border;
    

    
    
    
    img_raw = zeros(h*8, w*8);



    
    for ax = 1 : 8
        for ay = 1 : 8
            img_raw(ay:8:end, ax:8:end) = fullLF(:, :, ay, ax);
        end
    end
    
    input_img = zeros(h,w,2,2);

    
    input_img(:,:,1,1) = fullLF(:,:,2,2);
    input_img(:,:,1,2) = fullLF(:,:,2,7);
    input_img(:,:,2,1) = fullLF(:,:,7,2);
    input_img(:,:,2,2) = fullLF(:,:,7,7);
    
       
    img_1 = zeros(h,w,4);

    
    
    for ax = 1 : 2
        for ay = 1 : 2
            img_1( :, :, sub2ind([2 2], ax, ay)) = input_img(:, :, ay, ax);
        end
    end

    
    
    [H,W] = size(img_raw);
        
    for ix=1:floor(H/patch_size)
        for iy=1:floor(W/patch_size)
           patch_name = sprintf('D:/LF-E2E/data/train/%d',count);
           img_raw_patch =  img_raw( (ix-1)*patch_size + 1:ix * patch_size, (iy-1)*patch_size + 1:iy * patch_size);
           img_1_patch= img_1( (ix-1)*in_patch_size + 1:ix * in_patch_size, (iy-1)*in_patch_size + 1:iy * in_patch_size, :);
           patch = img_raw_patch;
           save(patch_name, 'patch');
           patch = img_1_patch;
           save(sprintf('%s_1', patch_name), 'patch');
           count = count+1;
           
        end
    end

    
    display(ns);
    
    
end
