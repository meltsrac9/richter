close all; clear all; clc;
layer = [3, 6, 9, 12, 16];
savePath = 'arcDataset/results/layerMaps';
imagePath = 'arcDataset/images';
% style = 'Palladian architecture';

style = 'Ancient Egyptian architecture';
% fileName = '3024_800px-United_States_Capitol_Building';
fileName = '1801_800px-Menkaure_Pyramid%2C_Giza%2C_Egypt6';
imageFileName = [imagePath '/' style '/' fileName '.jpg'];
img = imread(imageFileName);
imwrite(img,[savePath '/' fileName '.jpg'],'jpg','quality',100);
for l = layer;
    
    load(['arcDataset/data/cnn_whole_layer' num2str(l) '/' style '/' fileName '.mat'], 'cnnFeatures');
    
    A = figure;
    if size(cnnFeatures,1)<=100
        step = 1;
    else
        step = floor(size(cnnFeatures,1)/100);
    end
    count = 0;
    for x = 1:step:size(cnnFeatures,1);
        count = count+1;
        
        if count > 100
            break;
        end
        img = squeeze(cnnFeatures(x,:,:));
        subplot(10,10,count), imagesc(img);
        set(gca, 'XTick', []);
        set(gca, 'YTick', []);
        if count == 1
            title(num2str(step));
        end
    end
    saveas(A,[savePath '/' fileName '_layer' num2str(l) '.jpg'],'jpg');
end