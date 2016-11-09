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
startDataPath = 'arcDataset/data/SVMScores';
refDataPath = 'arcDataset/data/lbp';
load('arcDataset/folderList.mat','folderList');
styleNames = folderList(:,1);
imageCounts = cell2mat(folderList(:,2));
clear folderList;

%% Parameters
descriptor = 'cnn_whole_layer20';
descriptorSize = 25;
distance = 'euclidean';
k = 1;
savePath = ['arcDataset/results/' descriptor '_svmScores'];
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
    load([startDataPath '/' descriptor '_newsplit' num2str(split) '.mat'], 'allTrainScores', 'allTestScores');
    
    trainDescriptors = zeros(descriptorSize, sum(trainCounts));
    
    allImageData = cell(1,2);% Will store names of styles and image files here
    trainCounter = 0;
    counter = 0;
    fprintf('\nReading data...');
    for style = 1:size(testFileList,1)
        fprintf('%s...',styleNames{style});
        
        imagePath = [startImagePath '/' styleNames{style}];
        
        allNames = dir([refDataPath '/' styleNames{style} '/*.mat']);
        countWithinClass = 0;
        
        for x = 1:numel(allNames)
            counter = counter + 1;
            countWithinClass = countWithinClass + 1;
            
            
            if sum(countWithinClass==testFileList(style, :))==0
                trainCounter = trainCounter + 1;
                trainDescriptors(:,trainCounter) = allTrainScores(:,trainCounter);
                
                imageFileName = [imagePath '/' allNames(x).name(1:end-3) 'jpg'];
                if exist(imageFileName,'file')
                    allImageData{trainCounter, 2} = [imagePath '/' allNames(x).name(1:end-3) 'jpg'];
                else
                    allImageData{trainCounter, 2} = [imagePath '/' allNames(x).name(1:end-3) 'JPG'];
                end
                allImageData{trainCounter, 1} = styleNames{style};
                
            end
%             fprintf('\nCounter = %d, trainCounter = %d', counter, trainCounter);
        end
    end
    
    %     allDescriptors = allDescriptors - repmat(mean(allDescriptors),size(allDescriptors,1),1);
    %     allDescriptors = allDescriptors ./ repmat(std(allDescriptors),size(allDescriptors,1),1);
    fprintf('done.\nTesting...');
    %% Testing
    testData = zeros(descriptorSize, sum(testCounts),'single');
    testFileNums = zeros(1,sum(testCounts));
    count = 0;
    for style = 1:size(testFileList,1)
        for testFile = 1:size(testFileList,2)
            if testFileList(style,testFile)==0
                break;
            end
            count = count + 1;
            testFileNums(count) = sum(imageCounts(1:style-1))+ testFileList(style,testFile);
            testData(:,count) =  allTestScores(:,count);
        end
    end
    clear allTrainScores allTestScores;
    distances = pdist2(testData', trainDescriptors', distance);
    
    %% Classification
    for testFile = 1:numel(testFileNums)
        votes = zeros(1, numel(styleNames));
        testImageClassNum = 1;
        while sum(testCounts(1:testImageClassNum)) < testFile
            testImageClassNum = testImageClassNum + 1;
        end
        testImageClass = styleNames{testImageClassNum};
        [~, idx] = sort(distances(testFile,:));
        resultSetClasses = allImageData(idx(1:k), 1);
        for class = 1:numel(styleNames)
            match = strcmpi(styleNames{class},resultSetClasses);
            votes(class) = sum(match);
        end
        
        [~, md] = max(votes);
        resultClass = styleNames{md};
        if strcmpi(testImageClass,resultClass)
            tally(split,testFile) = 1;
        end
        
        conf(testImageClassNum,md,split) = conf(testImageClassNum,md,split) + 1;
        
        %% The next if block is for saving retrieval sets
        %         if mod(testFile,10)==0
        %             figure;
        %             queryImage = imread(allImageData{testFileNums(testFile), 2});
        %             imshow(queryImage);
        %
        %             imwrite(queryImage,[savePath '/' allImageData{testFileNums(testFile), 1} '_' descriptor '_'  num2str(testFile) '_query.jpg'],'jpg','quality',100);
        %             A = figure;
        %
        %             for x = 1:25
        %                 targetFile = allImageData{idx(x), 2};
        %                 image = imread(targetFile);
        %                 styleName = allImageData{idx(x), 1};
        %                 subplot(5,5,x), subimage(image);
        %                 set(gca, 'XTick', []);
        %                 set(gca, 'YTick', []);
        %                 title(styleName(1:5));
        %             end
        %             saveas(A,[savePath '/' allImageData{testFileNums(testFile), 1} '_' descriptor '_'  num2str(testFile) '_results.jpg'],'jpg')
        %             close all;
        %         end
        
    end
    fprintf('\nSuccess Rate: %.2f\n',sum(tally(split,:))/numel(tally(split,:)));
    total = total + sum(tally(split,:))/numel(tally(split,:));
    for style = 1:numel(styleNames)
        range = (sum(testCounts(1:style-1))+1):sum(testCounts(1:style));
        fprintf('%s: %.2f, ', styleNames{style}, sum(tally(split,range))/testCounts(style));
    end
    
    
    
    
    
end
save([savePath '/newConfusion.mat'], 'conf');

fprintf('\nDone!\n');
fprintf('\nFinal Success Rate: %.0f\n', 100*total / split);
for style = 1:numel(styleNames)
    range = (sum(testCounts(1:style-1))+1):sum(testCounts(1:style));
    total = 0;
    for split = 1:numSplits
        total = total + sum(tally(split, range))/testCounts(style);
    end
        fprintf('%s: %.2f\n', styleNames{style}, total / split);
%     fprintf('%.0f\n', 100*total / split);
    
end


