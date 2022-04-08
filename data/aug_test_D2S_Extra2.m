clearvars; clearvars -global; clc; close all; warning off;
sceneFolder = 'D:/LF_E2E/data/TrainingData/Test/30s';
count = 0;
list = dir([sceneFolder, '/*', []]);
dataNames = setdiff({list.name}, {'.', '..'});
dataPaths = strcat(strcat(sceneFolder, '/'), dataNames);
numDatasets = length(dataNames);
for ns = 1:numScenes

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
    
    fullLF = fullLF(:,:, 4:11, 4:11); % we only take the 8 middle images

    
    
    
    img_raw = zeros(h*8, w*8);



    
    for ax = 1 : 8
        for ay = 1 : 8
            img_raw(ay:8:end, ax:8:end) = fullLF(:, :, ay, ax);
        end
    end
    
    input_img = zeros(h,w,2,2);

    
    input_img(:,:,1,1) = fullLF(:,:,3,3);
    input_img(:,:,1,2) = fullLF(:,:,3,6);
    input_img(:,:,2,1) = fullLF(:,:,6,3);
    input_img(:,:,2,2) = fullLF(:,:,6,6);
    
       
    img_1 = zeros(h,w,4);
    
    
     for ax = 1 : 2
        for ay = 1 : 2
            img_1( :, :, sub2ind([2 2], ax, ay)) = input_img(:, :, ay, ax);
        end
    end

    
    
    [H,W] = size(img_raw);
    
    

    
    patch_name = sprintf('D:/LF_E2E/data/test/Set x/%d',count);
    save(patch_name, 'img_raw');
    save(sprintf('%s_1', patch_name), 'img_1');
    count = count+1;
    
    

    
    display(count);
    
    
end
