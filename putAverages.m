close all; clear all; fclose all;clc;

descriptor = 'cnn_box_layer16';
startImagePath = 'arcDataset/data/clusterImages_all';
% startDataPath1 = 'arcDataset/data/patchAll';
startDataPath1 = ['arcDataset/data/' descriptor];
startDataPath2 = 'arcDataset/data/clusterImages_all';
% make sure you use same spellings and case as folderList
load('arcDataset/folderList.mat');

filters = {'*.jpg'; '*.png'; '*.gif'; '*.bmp'; '*.tif'};
[fileName, pathName, filterIndex] = uigetfile(filters,'Select the query image');
queryImage = imread([pathName fileName]);
figure, imshow(queryImage);
load([startDataPath2 '/' descriptor '/clusters.mat'], 'patch');

pathComponents = strsplit(pathName,'/');
styleName = pathComponents{end-1};
queryImageDataFile = [startDataPath1 '/' styleName '/' fileName(1:end-3) 'mat' ];
load(queryImageDataFile, 'blocks');
desc = [];
for x = 1: size(blocks, 2)
    desc = [desc, blocks(x).cnnFeatures(:)];
end
desc = single(desc);
[clusterLabels, ~] = vl_kdtreequery(patch.kdtree, patch.clusters, desc);
newImage = 0*queryImage;
for x = 1: size(blocks, 2)
    imageFileName = [startImagePath '/' descriptor '/cluster_' num2str(clusterLabels(x)) '_mean.jpg'];
    patchImage = imread(imageFileName);
    width = blocks(x).cmax - blocks(x).cmin + 1;
    height = blocks(x).rmax - blocks(x).rmin + 1;
    patchImage = imresize(patchImage, [height, width]);
    newImage(blocks(x).rmin:blocks(x).rmax, blocks(x).cmin:blocks(x).cmax, :) = patchImage;
end
figure, imshow(newImage);
    