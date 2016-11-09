%% Overfeat CNN layer sizes
% layer 1: 96x56x56
% layer 2: 96x56x56
%% layer 3: 96x28x28
% layer 4: 256x24x24
% layer 5: 256x24x24
%% layer 6: 256x12x12
% layer 7:256x14x14
% layer 8: 512x12x12
%% layer 9: 512x12x12
% layer 10: 512x14x14
% layer 11: 1024x12x12
%% layer 12: 1024x12x12
% layer 13: 1024x14x14
% layer 14: 1024x12x12
% layer 15: 1024x12x12
%% layer 16: 1024x6x6
% layer 17: 3072x1x1
%% layer 18: 3072x1x1
% layer 19: 4096x1x1
%% layer 20: 4096x1x1


close all; clear all; fclose all;clc;

layer = 3;
startImagePath = 'arcDataset/images';
startDataPath = ['arcDataset/data/cnn_whole_layer' num2str(layer)];
% make sure you use same spellings and case as folderList
load('arcDataset/folderList.mat');

filters = {'*.jpg'; '*.png'; '*.gif'; '*.bmp'; '*.tif'};
[fileName, pathName, filterIndex] = uigetfile(filters,'Select the query image');
queryImage = imread([pathName fileName]);


queryDescriptor = overfeat(queryImage,layer,'whole'); 

querySum = queryDescriptor - min(queryDescriptor(:));
querySum = querySum / max(querySum(:));
querySum = squeeze(sum(querySum,1));
queryDescriptor = queryDescriptor(:);
querySum = querySum / max(querySum(:));
imageCounts = cell2mat(folderList(:,2));
querySum = imresize(querySum, [size(queryImage,1) size(queryImage,2)],'nearest');
newImage = uint8(single(queryImage) .* repmat(querySum, 1,1,3));
figure, imshow(newImage);
figure, imagesc(querySum);
allDescriptors = zeros(size(queryDescriptor,1),sum(imageCounts), 'single');

allImageData = cell(1,2);% Will store names of styles and image files here
count = 0;
for style = 1:numel(imageCounts)
    imagePath = [startImagePath '/' folderList{style,1}];
    dataPath = [startDataPath '/' folderList{style,1}];
    
    allNames = dir([dataPath '/*.mat']);
    
    for x = 1:numel(allNames)
        count = count + 1;
        inputFileName = [dataPath '/' allNames(x).name];
        allImageData{count, 1} = folderList{style,1};
        imageFileName = [imagePath '/' allNames(x).name(1:end-3) 'jpg'];
        if exist(imageFileName,'file')
            allImageData{count, 2} = [imagePath '/' allNames(x).name(1:end-3) 'jpg'];
        else
            allImageData{count, 2} = [imagePath '/' allNames(x).name(1:end-3) 'JPG'];
        end
        fprintf('%d\n',count);
        load(inputFileName,'cnnFeatures');
        allDescriptors(:,count) = cnnFeatures(:);
    end
end

distances = pdist2(queryDescriptor', allDescriptors');
[~, idx] = sort(distances);
fprintf('\nDone!\n');
blankImage = zeros([size(queryImage,1) size(queryImage,2) 3]);
figure;
for x = 1:25
    targetFile = allImageData{idx(x), 2};
    image = imread(targetFile);
    if size(image, 3) == 1
        image(:,:,2) = image;
        image(:,:,3) = image(:,:,2);
    end
    
    blankImage = blankImage + single(imresize(image, [size(queryImage,1) size(queryImage,2)]));
    styleName = allImageData{idx(x), 1};
    subplot(5,5,x), subimage(image);
    set(gca, 'XTick', []);
    set(gca, 'YTick', []);
    title(styleName(1:5));
end

blankImage = blankImage / max(blankImage(:));
blankImage = blankImage * 255;
blankImage = uint8(blankImage);
figure, imshow(blankImage);
figure, imshow(uint8(single(blankImage) .* repmat(querySum, 1,1,3)));



