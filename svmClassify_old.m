% Classify by SVM
clc; close all; clear all; fclose all;
% add required search paths

% --------------------------------------------------------------------
% Stage A: Data Preparation
% --------------------------------------------------------------------
startImagePath = 'arcDataset/images';
startDataPath = 'arcDataset/data';

load('arcDataset/folderList.mat','folderList');
styleNames = folderList(:,1);
imageCounts = cell2mat(folderList(:,2));
clear folderList;

%% Parameters
descriptor = 'cnn_whole_layer17';
descriptorSize = 3072;
distance = 'L1';
savePath = ['arcDataset/results/' descriptor];
makeDir([savePath '/']);
numTest = 30; % Number of test images in each class
numSplits = 5;
numClasses = numel(imageCounts);
splits = 5;
% Load data
testCount = 30;
testCounts = imageCounts * 0 + testCount;
trainCounts = imageCounts - 30;
count = 0;
grandTotal = 0;
for split = 1:splits
    fprintf('\nSplit %d.',split);
    load(['arcDataset/testFileList_' num2str(split) '.mat'], 'testFileList');
    
    trainDescriptors = zeros(descriptorSize,sum(trainCounts), 'single');
    testDescriptors = zeros(descriptorSize,testCount*numClasses, 'single');
    
    allImageData = cell(1,2);% Will store names of styles and image files here
    countTrain = 0;
    countTest = 0;
    fprintf('\nReading data...');
    for style = 1:size(testFileList,1)
        fprintf('%s...',styleNames{style});
        imagePath = [startImagePath '/' styleNames{style}];
        dataPath = [startDataPath '/' descriptor '/' styleNames{style} ];
        
        allNames = dir([dataPath '/*.mat']);
        countWithinClass = 0;
        for x = 1:numel(allNames)
            count = count + 1;

            inputFileName = [dataPath '/' allNames(x).name];
            allImageData{count, 1} = styleNames{style};
            imageFileName = [imagePath '/' allNames(x).name(1:end-3) 'jpg'];
            if exist(imageFileName,'file')
                allImageData{count, 2} = [imagePath '/' allNames(x).name(1:end-3) 'jpg'];
            else
                allImageData{count, 2} = [imagePath '/' allNames(x).name(1:end-3) 'JPG'];
            end
            load(inputFileName);
            countWithinClass = countWithinClass + 1;
                        
            if strcmpi(descriptor(1:3),'cnn')
                descriptorType = 'cnn';
            else
                descriptorType = descriptor;
            end
            switch descriptorType
                case 'cnn'
                    tempDescriptor = cnnFeatures(:);
                case 'lbp'
                    tempDescriptor = LBP;
                case 'tinyIm'
                    tempDescriptor = tinyIm;
                case 'colorHist'
                    tempDescriptor = colorHist;
                case 'colorlbp'
                    tempDescriptor = colorLBP;
                otherwise
                    error(['Invalid descriptor: ' descriptor])
            end
            if sum(countWithinClass==testFileList(style, :))>0
                countTest = countTest + 1;                            
                testDescriptors(:,countTest) = tempDescriptor;
            else
                countTrain = countTrain + 1;
                trainDescriptors(:,countTrain) = tempDescriptor;
            end
        end        
    end
    %     allDescriptors = allDescriptors - repmat(mean(allDescriptors),size(allDescriptors,1),1);
    %     allDescriptors = allDescriptors ./ repmat(std(allDescriptors),size(allDescriptors,1),1);
    fprintf('done.\nTesting...');
    
%     keyboard;
    
    
    
    
    
    
    successRate = 0;
    

    
    histograms = trainDescriptors;
    testHistograms = testDescriptors ;
    clear trainData testData;
    allScores = zeros(numClasses, sum(testCounts));
    realClasses = zeros(1, sum(testCounts));
   
    % Optional: Vary the classifier (Hellinger kernel)
%     histograms = histograms - repmat(min(histograms),descriptorSize, 1);
%     histograms = histograms ./ repmat(max(histograms),descriptorSize, 1);
%     testHistograms = testHistograms - repmat(min(testHistograms),descriptorSize, 1);
%     testHistograms = testHistograms ./ repmat(max(testHistograms),descriptorSize, 1);
%     histograms = sqrt(histograms);
%     testHistograms = sqrt(testHistograms);
    
    % Optional: L2 normalize the histograms before running the linear SVM
%     histograms = bsxfun(@times, histograms, 1./sqrt(sum(histograms.^2,1))) ;
%     testHistograms = bsxfun(@times, testHistograms, 1./sqrt(sum(testHistograms.^2,1))) ;
    fprintf('Split %d: ', split) ;
    
    for style = 1:numClasses
        fprintf('%d ', style) ;
        noTrain = trainCounts(style);
        noTest = testCounts(style);
        % Prepare training labels
        labels = [- ones(1,sum(trainCounts(1:(style-1)))), ones(1,noTrain), - ones(1,sum(trainCounts((style+1):end)))] ;
        
        % Prepare testing labels
        testLabels =  [- ones(1,sum(testCounts(1:(style-1)))), ones(1,noTest), - ones(1,sum(testCounts((style+1):end)))] ;
        realClasses(sum(testCounts(1:(style-1)))+1:sum(testCounts(1:style))) = style;
        
        % count how many images are there
        %     fprintf('Number of training images: %d positive, %d negative\n', ...
        %         sum(labels > 0), sum(labels < 0)) ;
        %     fprintf('Number of testing images: %d positive, %d negative\n', ...
        %         sum(testLabels > 0), sum(testLabels < 0)) ;
        
        
        % --------------------------------------------------------------------
        % Stage B: Training a classifier
        % --------------------------------------------------------------------
        
        % Train the linear SVM
        C = 10 ;
        [w, bias] =  vl_svmtrain(histograms, labels, C) ;
        
        % Evaluate the scores on the training data
        scores = w' * histograms + bias ;
        
        % Visualize the ranked list of images
        % figure(1) ; clf ; set(1,'name','Ranked training images (subset)') ;
        % displayRankedImageList(names, scores)  ;
        
        % Visualize the precision-recall curve
%             figure(2) ; clf ; set(2,'name','Precision-recall on train data') ;
%             vl_pr(labels, scores) ;
        
        % --------------------------------------------------------------------
        % Stage C: Classify the test images and assess the performance
        % --------------------------------------------------------------------
        
        % Test the linar SVM
        testScores = w' * testHistograms + bias ;
        
        allScores(style, :) = testScores - mean(testScores);
        allScores(style, :) = allScores(style, :) / std(allScores(style, :));
        % Visualize the ranked list of images
        % figure(3) ; clf ; set(3,'name','Ranked test images (subset)') ;
        % displayRankedImageList(testNames, testScores)  ;
        
        % Visualize the precision-recall curve
%             figure(4) ; clf ; set(4,'name','Precision-recall on test data') ;
%             vl_pr(testLabels, testScores) ;
        
%         keyboard;
        % Print results
        [~,~,info] = vl_pr(labels, scores) ;
        %     fprintf('Test AP: %.2f\n', info.auc) ;
        successRate = successRate + info.auc;
        
%         [~,perm] = sort(testScores,'descend') ;
%         fprintf('Correctly retrieved in the top %d: %d\n', noTest, sum(testLabels(perm(1:noTest)) > 0)) ;
    end
    fprintf('\nTest AP: %.4f\n', successRate/numClasses) ;
    
    [~, guesses] = max(allScores);
    myRate = sum(guesses==realClasses)/sum(testCounts);
    fprintf('\nMy AP: %.4f\n', myRate) ;
    grandTotal = grandTotal + myRate;

   
end

fprintf('\nMean AP: %.4f\n', grandTotal/splits) ;

