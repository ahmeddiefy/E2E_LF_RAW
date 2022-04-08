clearvars; clearvars -global; clc; close all; warning off;
sceneFolder = 'D:/LF_E2E/data/TrainingData/Test/30scenes';
usFolder = 'D:/LF_E2E/results/numpy';

list = dir([sceneFolder, '/*', []]);
dataNames = setdiff({list.name}, {'.', '..'});
dataPaths = strcat(strcat(sceneFolder, '/'), dataNames);
numDatasets = length(dataNames);
for ns = 1:numScenes
    count = 1;
    numImgsX = 14;
    numImgsY = 14;
    folder_name = sceneNames{ns};

    resultPath = [scenePaths{ns}];
    mkdir(strcat('D:/LF_E2E/checkpoints/2 mdpi paper/US/occ/',folder_name(1:end-4)));
%%% converting the extracted light field to a different format
    inputImg = im2double(imread(resultPath));
    inputImg = rgb2ycbcr(inputImg);

    h = size(inputImg, 1) / numImgsY;
    w = size(inputImg, 2) / numImgsX;

    fullLF = zeros(h, w, 3, numImgsY, numImgsX);

    for ax = 1 : numImgsX
        for ay = 1 : numImgsY
            fullLF(:, :, :, ay, ax) = inputImg(ay:numImgsY:end, ax:numImgsX:end, :);
        end
    end

    fullLF = fullLF(23:h-22, 23:w-22, :, 4:11, 4:11); % we only take the 8 middle images
    h=h-2*22;
    w=w-2*22;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    input_small = zeros(2,2,h,w,3);
    input_small_1 = zeros(2,8,h,w,3);
    input_small_2 = zeros(8,8,h,w,3);

    
    input_small(1,1,:,:,:) = fullLF(:,:,:,1,1);
    input_small(1,2,:,:,:) = fullLF(:,:,:,1,8);
    
    input_small(2,1,:,:,:) = fullLF(:,:,:,8,1);
    input_small(2,2,:,:,:) = fullLF(:,:,:,8,8);

   
    
     for i = 1:h
        for v=1:2
          input_small_1(v,:,i,:,:) = imresize(squeeze(input_small(v,:,i,:,:)),[8 w]);   
        end
    end
    
    for i = 1:w
        for v=1:8
          input_small_2(:,v,:,i,:) = imresize(squeeze(input_small_1(:,v,:,i,:)),[8 h]);   
        end
    end
    
    
   image_ycbcr=single(permute(input_small_2,[3,4,5,1,2]));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   out = load(strcat(usFolder,num2str(ns-1),'.mat'));
   out=out.a;
   [ho,wo]= size(out);
   hh=ho/8;
   ww=wo/8;
    
   out_new=zeros(hh,ww,8,8);
    
    for ax = 1 : 8
        for ay = 1 : 8
            out_new(:, :, ay, ax) = out(ay:8:end, ax:8:end);
        end
    end
    out_new = out_new(23:hh-22, 23:ww-22, :, :);
    image_ycbcr(:,:,1,:,:)=out_new;
    
    
    for ax = 1 : 8
        for ay = 1 : 8
            imwrite(ycbcr2rgb(squeeze(image_ycbcr(:, :, :, ax, ay))),strcat('D:/LF_E2E/results/numpy/30scenes/',folder_name(1:end-4),'/',num2str(count),'.png'))
            count = count+1;

        end
    end

end