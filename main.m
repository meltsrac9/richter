close all; clear all; fclose all;clc;
filters = {'*.jpg'; '*.png'; '*.gif'; '*.bmp'; '*.tif'};
[fileName, pathName, filterIndex] = uigetfile(filters,'Select the query image');
image = imread([pathName fileName]);
figure, imshow(image);
queryData = single(colorHistogram(image));

imagePath = '/home/banerji/work/misc/VOC2012/JPEGImages';% Folder where images are
dataPath = '/home/banerji/work/misc/VOC2012/Data/Descriptors/ColorHist'; %Folder for storing data
allNames = dir([imagePath '/*.jpg']);
targetData = zeros(size(queryData,1),numel(allNames),'single');
makeDir([dataPath '/']);
for x = 1:numel(allNames)
        saveFile = [dataPath '/' allNames(x).name(1:end-3) 'mat'];

    %% comment in the next four lines and comment out the following two 
    %% if you are running it for the first time. On subsequent runs, 
    %% comment out the next four and make the following two active
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    targetFile = [imagePath '/' allNames(x).name];
%     targetImage = imread(targetFile);
%     colorHist = colorHistogram(targetImage);
%     save(saveFile,'colorHist');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    load(saveFile, 'colorHist');
    targetData(:,x) = colorHist;
    fprintf('\n%s',saveFile);
end
distances = pdist2(queryData', targetData','cosine');
[~, idx] = sort(distances);
fprintf('\nDone!\n');
figure;
for x = 1:49
    targetFile = [imagePath '/' allNames(idx(x)).name];
    image = imread(targetFile);
    subplot(7,7,x), subimage(image);
    set(gca, 'XTick', []);
    set(gca, 'YTick', []);
end

