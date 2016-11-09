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

startImagePath = 'arcDataset/images';
startDataPath = 'arcDataset/data/clusterImages_all';
refDataPath = 'arcDataset/data/cnn_box_layer20';
load('arcDataset/folderList.mat','folderList');
styleNames = folderList(:,1);
imageCounts = cell2mat(folderList(:,2));
clear folderList;

%% Parameters
descriptor = 'cnn_box_layer20';
descriptorSize = 4096;
distance = 'euclidean';
k = 1;
savePath = ['arcDataset/results/' descriptor '_BOW'];
makeDir([savePath '/']);
numSplits = 5;
numClasses = numel(imageCounts);

trainCount = 30;% Number of train images in each class
testCounts = imageCounts - trainCount;
trainCounts = imageCounts*0 + 30;
%%
tally = zeros(5,sum(testCounts));

total = 0;
conf = zeros(numel(styleNames),numel(styleNames),5);
fprintf('\nDescriptor: %s, k = %d\n',descriptor,k);
%% Reading
for split = 1:numSplits
    fprintf('\nSplit %d.',split);
    load(['arcDataset/testFileList_' num2str(split) '.mat'], 'testFileList');
    load([startDataPath '/' descriptor '/clusters.mat'], 'patch');
    load([startDataPath '/' descriptor '/allData.mat'], 'bigDesc', 'allPaths');
    testDescriptors = zeros(size(bigDesc, 1), sum(testCounts), 'single');
    
    allImageData = cell(1,2);% Will store names of styles and image files here
    testCounter = 0;
    counter = 0;
    fprintf('\nReading data...');
    testImageBoxCounts = [];
    for style = 1:size(testFileList,1)
        fprintf('%s...',styleNames{style});
        
        imagePath = [startImagePath '/' styleNames{style}];
        
        allNames = dir([refDataPath '/' styleNames{style} '/*.mat']);
        countWithinClass = 0;
        
        for x = 1:numel(allNames)
            
            countWithinClass = countWithinClass + 1;
            load([refDataPath '/' styleNames{style} '/' allNames(x).name],'boxes');
            numBoxes = size(boxes, 1);
            clear boxes;
            
            if sum(countWithinClass==testFileList(style, :))>0
                testImageBoxCounts = [testImageBoxCounts; numBoxes];
                for b = 1:numBoxes
                    counter = counter + 1;
                    testCounter = testCounter + 1;
                    testDescriptors(:,testCounter) = bigDesc(:,counter);
                    
                    
                    imageFileName = allPaths{counter, 2};
                end
            else
                counter = counter + numBoxes;
            end
        end
    end
    
    %     allDescriptors = allDescriptors - repmat(mean(allDescriptors),size(allDescriptors,1),1);
    %     allDescriptors = allDescriptors ./ repmat(std(allDescriptors),size(allDescriptors,1),1);
    fprintf('done.\nTesting...');
    %% Testing
    testFileNums = zeros(1,sum(testCounts));
    count = 0;
    for style = 1:size(testFileList,1)
        for testFile = 1:size(testFileList,2)
            if testFileList(style,testFile)==0
                break;
            end
            count = count + 1;
            testFileNums(count) = sum(imageCounts(1:style-1))+ testFileList(style,testFile);
        end
    end
    
    [testClusterLabels, dist] = vl_kdtreequery(patch.kdtree, patch.clusters, testDescriptors, 'numneighbors', size(patch.clusters,2));
    %% Classification
    testData = zeros(size(patch.clusters,2), numel(testFileNums), 'single');
    for testFile = 1:numel(testFileNums)
        start = sum(testImageBoxCounts(1:testFile-1))+1;
        stop = sum(testImageBoxCounts(1:testFile));
        for c = start:stop
            testData(testClusterLabels(:,c), testFile) = testData(testClusterLabels(:,c), testFile) + (1000/dist(:,c))';
        end 
    end
    keyboard;
end


