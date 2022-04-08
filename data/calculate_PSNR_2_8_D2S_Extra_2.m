clearvars; clearvars -global; clc; close all; warning off;
sceneFolder = 'D:/LF_E2E/results/numpy/';

list = dir([sceneFolder, '/*', []]);
dataNames = setdiff({list.name}, {'.', '..'});
dataPaths = strcat(strcat(sceneFolder, '/'), dataNames);
numDatasets = length(dataNames);

out_psnr=0;
out_ssim=0;
PSNR_Without_in = 0;
SSIM_Without_in = 0;

number = numScenes/2;


for ns = 1:number

    display(ns)
    gt = load(strcat(sceneFolder,num2str(ns-1),'_gt.mat'));
    out = load(strcat(sceneFolder,num2str(ns-1),'.mat'));
    gt=single(gt.a);
    out=out.a;
    
    
    [h,w]= size(gt);
    hh=h/8;
    ww=w/8;
    
    gt_new=zeros(hh,ww,8,8);
    out_new=zeros(hh,ww,8,8);
    
    for ax = 1 : 8
        for ay = 1 : 8
            gt_new(:, :, ay, ax) = gt(ay:8:end, ax:8:end);
            out_new(:, :, ay, ax) = out(ay:8:end, ax:8:end);
        end
    end
    
    outPSNR = zeros(8,8);
    outSSIM = zeros(8,8);
    
    for ax = 1 : 8
        for ay = 1 : 8

                outPSNR(ay,ax) = psnr(CropImg(squeeze(out_new(:,:,ay,ax)),22), CropImg(squeeze(gt_new(:,:,ay,ax)),22));
                outSSIM(ay,ax) = ssim(uint8(255*CropImg(squeeze(out_new(:,:,ay,ax)),22)), uint8(255*CropImg(squeeze(gt_new(:,:,ay,ax)),22)));

        end
    end
    
    outPSNR
    


    outPSNR;
    fprintf('PSNR average: ');
%     sum(sum(outPSNR))/(ang_resolution*ang_resolution)
    fprintf('PSNR average without input:\n');
    outPSNR(3,3) = 0;
    outPSNR(3,6) = 0;
    outPSNR(6,3) = 0;
    outPSNR(6,6) = 0;
    psnr_without = sum(sum(outPSNR))/60

    

    out_psnr = out_psnr + outPSNR;
    PSNR_Without_in = PSNR_Without_in + psnr_without;

    
    outSSIM;
    fprintf('PSNR average: ');
%     sum(sum(outPSNR))/(ang_resolution*ang_resolution)
    fprintf('PSNR average without input:\n');
    outSSIM(3,3) = 0;
    outSSIM(3,6) = 0;
    outSSIM(6,3) = 0;
    outSSIM(6,6) = 0;
    ssim_without = sum(sum(outSSIM))/60

    

    out_ssim = out_ssim + outSSIM;
    SSIM_Without_in = SSIM_Without_in + ssim_without;

    
end
fprintf('Summary:  average:\n');

fprintf('PSNR average:\n');
out_psnr = out_psnr/number
sum(sum(out_psnr))/(8*8)

fprintf('PSNR average without input:\n');
sum(sum(PSNR_Without_in))/number


fprintf('SSIM average:\n');
out_ssim = out_ssim/number
sum(sum(out_ssim))/(8*8)

fprintf('SSIM average without input:\n');
sum(sum(SSIM_Without_in))/number