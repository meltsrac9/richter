close all; clear all; fclose all;clc;

startImagePath = 'arcDataset/images';
startDataPath = ['arcDataset/data/SVMScores'];

% make sure you use same spellings and case as folderList
load('arcDataset/folderList.mat');
styleNames = folderList(:,1);
imageCounts = cell2mat(folderList(:,2));
clear folderList;
distance = 'L1';
k = 1;

for layer = 1:20
    %% Parameters
    descriptor = ['cnn_whole_layer' num2str(layer)];
    descriptorSize = 25;
    distance = 'Euclidean';
    savePath = ['arcDataset/results/' descriptor '_svmScores'];
    makeDir([savePath '/']);
    numTest = 30; % Number of test images in each class
    numSplits = 5;
    numClasses = numel(imageCounts);
    testCount = 30;
    testCounts = imageCounts * 0 + testCount;
    trainCounts = imageCounts - 30;
    for split = 1:numSplits
        load(['arcDataset/testFileList_' num2str(split) '.mat']);
        
        load([startDataPath '/' descriptor '_split' num2str(split) '.mat'],'allTrainScores','allTestScores');
        for testClass = 1:numClasses
            for testFile = 1:testCounts(testClass)
                testFileNum = sum(testCounts(1:testClass-1))+ testFile;
                queryDescriptor = allTestScores(:,testFileNum);
                
                if testClass==1 && testFile==1
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
%                                 if mod(find(countWithinClass==testFileList(style,:)),10)==0
%                                     fileName = [savePath '/split' num2str(split) '_class' num2str(style) '_query' num2str(countWithinClass) '.jpg'];
%                                     if ~exist(fileName,'file')
%                                         img = imread(imageFileName);
%                                         figure, imshow(img);
%                                         imwrite(img, fileName,'jpg','quality',100);
%                                         close all;
%                                     end
%                                 end
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
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %% Classification
                for testF = 1:numel(testFileList)
                votes = zeros(1, numel(styleNames));
                testImageClass = allImageData{testFileNums(testF), 1};
        testImageClassNum = find(strcmpi(testImageClass, styleNames));
        % To remove query image from retrieval list
        distances(testFile,testFileNums(testFile)) = inf;
        
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
        range = ((style-1)*numTest+1):(style*numTest);
        fprintf('%s: %.2f, ', styleNames{style}, sum(tally(split,range))/numTest);
    end
    
 
                
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                fprintf('\nQuery %d of class %d done!\n', testFile, testClass);
%                 A = figure;
%                 for x = 1:25
%                     targetFile = allImageData{idx(x), 2};
%                     image = imread(targetFile);
%                     styleName = allImageData{idx(x), 1};
%                     vl_tightsubplot(5,5,x); subimage(image);
%                     set(gca, 'XTick', []);
%                     set(gca, 'YTick', []);
%                     %                 title(styleName(1:5));
%                 end
%                 saveas(A, [savePath '/split' num2str(split) '_class' num2str(testClass) '_query' num2str(testFileList(testClass,testFile)) '_result.jpg'],'jpg');
%                 close all;
            end
        end
    end
end




