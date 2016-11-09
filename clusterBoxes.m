close all; clear all; fclose all;clc;


startImagePath = 'arcDataset/images';
startDataPath = 'arcDataset/data';
load('arcDataset/folderList.mat','folderList');
styleNames = folderList(:,1);
imageCounts = cell2mat(folderList(:,2));
clear folderList;

descriptors = {'lbp','patchHOG','tinyIm','colorHist','colorLBP'};
descriptorSizes = [256, 496, 768, 768, 768];
numClusters = 50;

distance = 'L1';
for descriptor = 1:5
    bigDesc = [];
    bigNames = [];
    allPaths = cell(1,2);
    imageCount = 0;
    for style = 1:numel(styleNames)
        imagePath = [startImagePath '/' styleNames{style}];
        dataPath = [startDataPath '/patchAll/' styleNames{style} ];
        dataPath2 = [startDataPath '/patchImages/' styleNames{style} ];
        
        savePath = [startDataPath '/clusterImages_all/' descriptors{descriptor} ];
        
        allNames = dir([imagePath '/*.*']);
        desc = [];
        
        
        makeDir([savePath '/']); %creates the data folder if it doesn't exist
        for f = 1:numel(allNames)
            if strcmp(allNames(f).name, '.') || strcmp(allNames(f).name, '..')
                continue;
            end
            
            load([startDataPath '/patchAll/' styleNames{style} '/' allNames(f).name(1:end-3) 'mat'],'blocks');
            fprintf([startDataPath '/patchAll/' styleNames{style} '/' allNames(f).name(1:end-3) 'mat\n']);
            for b = 1:size(blocks,2)
                imageCount = imageCount + 1;
                patchFile = [dataPath2 '/' allNames(f).name(1:end-4) '_' num2str(b) '.jpg'];
                allPaths{imageCount, 2} = patchFile;
                allPaths{imageCount, 1} = styleNames{style};
            
                switch descriptor
                    case 1
                        desc = [desc, blocks(b).lbp(:)];
                    case 2
                        desc = [desc, blocks(b).patchHOG(:)];
                    case 3
                        desc = [desc, blocks(b).tinyIm(:)];
                    case 4
                        desc = [desc, blocks(b).colorHist(:)];
                    otherwise
                        desc = [desc, blocks(b).colorLBP(:)];
                end
            end
        end
        bigDesc = [bigDesc, desc];
        bigNames = [bigNames; allNames];
    end
    fprintf(['Computing patch clusters and kdtree for descriptor ' descriptors{descriptor} '\n']) ;
    bigDesc = single(bigDesc);
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
            title([fewNames{y,1}(1:2) '_' fewNames{y,2}(end-6:end-5)]);
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
    fprintf(['\nEnd of descriptor ' descriptors{descriptor} '!\n'] ) ;
    
end
