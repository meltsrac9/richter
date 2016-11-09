close all; clear all; fclose all;clc;


startImagePath = 'arcDataset/images';
startDataPath = 'arcDataset/data';
load('arcDataset/folderList.mat','folderList');
styleNames = folderList(:,1);
imageCounts = cell2mat(folderList(:,2));
clear folderList;
layers = [3, 6, 9, 12, 16, 18, 20];
descriptorName = 'cnn_box_layer';
descriptorSizes = [75264, 36864, 73728, 147456, 36864, 3072, 4096];
numClusters = 1000;

distance = 'L1';
load(['arcDataset/testFileList_1.mat'], 'testFileList');
for descriptor = 1:numel(layers)
    bigDesc = [];
    bigNames = [];
    allPaths = cell(1,2);
    imageCount = 0;
    for style = 1:numel(styleNames)
        imagePath = [startImagePath '/' styleNames{style}];
        dataPath = [startDataPath '/' descriptorName num2str(layers(descriptor)) '/' styleNames{style} ];
        dataPath2 = [startDataPath '/patchImages/' styleNames{style} ];
        
        savePath = [startDataPath '/clusterImages_all/' descriptorName num2str(layers(descriptor)) '_1000'];
        
        allNames = dir([imagePath '/*.*']);
        desc = [];
        
        
        makeDir([savePath '/']); %creates the data folder if it doesn't exist
        countWithinClass = 0;

        for f = 1:numel(allNames)
            if strcmp(allNames(f).name, '.') || strcmp(allNames(f).name, '..')
                continue;
            end
            countWithinClass = countWithinClass + 1;
            if sum(countWithinClass==testFileList(style, :))==0
                load([startDataPath '/' descriptorName num2str(layers(descriptor)) '/' styleNames{style} '/' allNames(f).name(1:end-3) 'mat'],'blocks');
                fprintf([startDataPath '/' descriptorName num2str(layers(descriptor)) '/' styleNames{style} '/' allNames(f).name(1:end-3) 'mat\n']);
                for b = 1:size(blocks,2)
                    imageCount = imageCount + 1;
                    
                    patchFile = [dataPath2 '/' allNames(f).name(1:end-4) '_' num2str(b) '.jpg'];
                    allPaths{imageCount, 2} = patchFile;
                    allPaths{imageCount, 1} = styleNames{style};
                    
                    desc = [desc , blocks(b).cnnFeatures(:)];
                end
            end
        end
        bigDesc = [bigDesc, desc];
        if size(bigDesc,2) ~= size(allPaths,1)
            keyboard;
        end
        bigNames = [bigNames; allNames(3:end)];
    end
    save([savePath '/allData.mat'], 'bigDesc', 'allPaths');
    fprintf(['\nComputing patch clusters and kdtree for descriptor ' descriptorName num2str(layers(descriptor)) '\n']) ;
    patch.clusters = vl_kmeans(bigDesc, numClusters, 'algorithm', 'elkan') ;
    patch.kdtree = vl_kdtreebuild(patch.clusters) ;
    
    
    centers = patch.clusters';
    %         keyboard;
    distances = zeros(size(bigDesc,2),1);
    [clusterLabels, ~] = vl_kdtreequery(patch.kdtree, patch.clusters, bigDesc);
    styleVotes = zeros(1, numel(styleNames));
    mainStyleNums = zeros(1,numClusters);
    for x = 1:numClusters
        imagesWithinCluster = (clusterLabels==x);
        fewNames = allPaths(imagesWithinCluster,1);
        for y = 1:size(fewNames,1)
            styleVotes = styleVotes + (strcmpi(fewNames{y,1},styleNames)/numel(imagesWithinCluster))';
        end
        [~,mainStyleNums(x)] = max(styleVotes);
    end
    
    save([savePath '/clusters.mat'],'patch','mainStyleNums');
    %     keyboard;
    top = [5,5];
    
    for x = 1:numClusters
        imagesWithinCluster = (clusterLabels==x);
        fewNames = allPaths(imagesWithinCluster,:);
        A = figure;
        for y = 1:min([top(1)*top(2), sum(imagesWithinCluster)])
            image = imread(fewNames{y,2});
            subplot(top(1),top(2),y), subimage(image);
            set(gca, 'XTick', []);
            set(gca, 'YTick', []);
            title([fewNames{y,1}(1:3) '_' fewNames{y,2}(end-6:end-5)]);
        end
        saveas(A,[savePath '/cluster_'  num2str(x) '.jpg'],'jpg');
        blankImage = zeros(256, 256, 3);
        for y = 1:sum(imagesWithinCluster)
            image = imread(fewNames{y,2});
            
            if size(image, 3) == 1
                image(:,:,2) = image;
                image(:,:,3) = image(:,:,2);
            end
            blankImage = blankImage + single(imresize(image, [256, 256]));
        end
        blankImage = blankImage - min(blankImage(:));
        blankImage = blankImage / max(blankImage(:));
        blankImage = blankImage * 255;
        blankImage = uint8(blankImage);
        figure, imshow(blankImage);
        imwrite(blankImage, [savePath '/cluster_'  num2str(x) '_mean.jpg'],'jpg');
        close all;
        
        
    end
    fprintf(['\nEnd of descriptor ' descriptorName num2str(layers(descriptor)) '!\n'] ) ;
    
end
