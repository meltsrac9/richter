close all; clear all; fclose all;clc;

layer = 5;
startImagePath = 'arcDataset/images';
startDataPath = ['arcDataset/data/cnn_whole_layer' num2str(layer)];
% make sure you use same spellings and case as folderList
load('arcDataset/folderList.mat');

filters = {'*.jpg'; '*.png'; '*.gif'; '*.bmp'; '*.tif'};
[fileName, pathName, filterIndex] = uigetfile(filters,'Select the query image');
queryImage = imread([pathName fileName]);
figure, imshow(queryImage);

queryDescriptor = overfeat(queryImage,layer,'whole'); 
queryDescriptor = queryDescriptor(:);
imageCounts = cell2mat(folderList(:,2));

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
figure;
for x = 1:25
    targetFile = allImageData{idx(x), 2};
    image = imread(targetFile);
    styleName = allImageData{idx(x), 1};
    subplot(5,5,x), subimage(image);
    set(gca, 'XTick', []);
    set(gca, 'YTick', []);
    title(styleName(1:5));
end



