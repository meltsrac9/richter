close all; clear all; fclose all;clc;

startImagePath = 'arcDataset/images';
startDataPath = ['arcDataset/data/SVMScores'];

% make sure you use same spellings and case as folderList
load('arcDataset/folderList.mat');
styleNames = folderList(:,1);
imageCounts = cell2mat(folderList(:,2));
clear folderList;

for layer = 1:20
    %% Parameters
    descriptor = ['cnn_whole_layer' num2str(layer)];
    descriptorSize = 25;
    distance = 'Euclidean';
    savePath = ['arcDataset/results/' descriptor '_svmScores'];
    makeDir([savePath '/']);
    numTest = 30; % Number of test images in each class
    numSplits = 1;
    numClasses = numel(imageCounts);
    testCount = 30;
    testCounts = imageCounts * 0 + testCount;
    trainCounts = imageCounts - testCount;
    for split = 1:numSplits
        load(['arcDataset/testFileList_' num2str(split) '.mat']);
        
        load([startDataPath '/' descriptor '_split' num2str(split) '.mat'],'allTrainScores','allTestScores');
        for testClass = 1:numClasses
            for testFile = 10:10:testCounts(testClass) %1:testCounts(testClass)
                testFileNum = sum(testCounts(1:testClass-1))+ testFile;
                queryDescriptor = allTestScores(:,testFileNum);
                
                if testClass==1 && testFile==10 %1
                    
                    trainDescriptors = zeros(descriptorSize, sum(trainCounts));
                    allImageData = cell(1,2);% Will store names of styles and image files here
                    trainCounter = 0;
                    counter = 0;
                    for style = 1:numel(imageCounts)
                        fprintf('\nStyle = %d', style);
                        
                        imagePath = [startImagePath '/' styleNames{style}];
                        
                        allNames = dir([imagePath '/*.*']);
                        countWithinClass = 0;
                        for x = 1:numel(allNames)
                            if strcmp(allNames(x).name, '.') || strcmp(allNames(x).name, '..')
                                continue;
                            end
                            counter = counter + 1;
                            imageFileName = [imagePath '/' allNames(x).name];
                            countWithinClass = countWithinClass + 1;
                            
                            if sum(countWithinClass==testFileList(style, :))>0
                                if mod(find(countWithinClass==testFileList(style,:)),10)==0
                                    fileName = [savePath '/split' num2str(split) '_class' num2str(style) '_query' num2str(countWithinClass) '.jpg'];
                                    if ~exist(fileName,'file')
                                        img = imread(imageFileName);
                                        figure, imshow(img);
                                        imwrite(img, fileName,'jpg','quality',100);
                                        close all;
                                    end
                                end
                            else
                                
                                trainCounter = trainCounter + 1;
                                trainDescriptors(:,trainCounter) = allTrainScores(:,trainCounter);
                                allImageData{trainCounter, 1} = styleNames{style};
                                allImageData{trainCounter, 2} = imageFileName;
                            end
                            fprintf('\nCounter = %d, trainCounter = %d', counter, trainCounter);
                        end
                    end
                    
                end
                distances = pdist2(queryDescriptor', trainDescriptors');
                [~, idx] = sort(distances);
                fprintf('\nQuery %d of class %d done!\n', testFile, testClass);
                A = figure;
                for x = 1:25
                    targetFile = allImageData{idx(x), 2};
                    image = imread(targetFile);
                    styleName = allImageData{idx(x), 1};
                    vl_tightsubplot(5,5,x); subimage(image);
                    set(gca, 'XTick', []);
                    set(gca, 'YTick', []);
                    %                 title(styleName(1:5));
                end
                saveas(A, [savePath '/split' num2str(split) '_class' num2str(testClass) '_query' num2str(testFileList(testClass,testFile)) '_result.jpg'],'jpg');
                close all;
            end
        end
    end
end




