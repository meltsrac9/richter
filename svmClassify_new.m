% Classify by SVM
clc; close all; clear all; fclose all;
% add required search paths

% --------------------------------------------------------------------
% Stage A: Data Preparation
% --------------------------------------------------------------------

load('arcDataset/folderList.mat','folderList');
styleNames = folderList(:,1);
imageCounts = cell2mat(folderList(:,2));
clear folderList;
startImagePath = 'arcDataset/images';
startDataPath = 'arcDataset/data';
savePath = 'arcDataset/data/SVMScores';
makeDir([savePath '/']);

for layer = 9:9
    %% Parameters
    descriptor = ['cnn_whole_layer' num2str(layer)];
    tempImage = imread('football.jpg');
    tempCNN = overfeat(tempImage, layer, 'whole');
    descriptorSize = numel(tempCNN);
    clear tempImage tempCNN;
%     descriptor = 'tinyIm';
%     descriptorSize = 768;
    numClasses = numel(imageCounts);
    splits = 5;
    % Load data
    trainCount = 30;% Number of train images in each class
    testCounts = imageCounts - trainCount;
    trainCounts = imageCounts*0 + 30;
    count = 0;
    grandTotal = 0;
    for split = 1:splits
        fprintf('\nSplit %d.',split);
        load(['arcDataset/testFileList_' num2str(split) '.mat'], 'testFileList');
        
        trainDescriptors = zeros(descriptorSize,sum(trainCounts), 'single');
        testDescriptors = zeros(descriptorSize,sum(testCounts), 'single');
        
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
        fprintf('done.\n');
        
        %     keyboard;
        
        successRate = 0;
        
        allTrainScores = zeros(numClasses, sum(trainCounts));
        allTestScores = zeros(numClasses, sum(testCounts));
        realClasses = zeros(1, sum(testCounts));
        
        fprintf('Split %d: ', split) ;
        
        for style = 1:numClasses
            fprintf('%d ', style) ;
            noTrain = trainCounts(style);
            noTest = testCounts(style);
            % Prepare training trainLabels
            trainLabels = [- ones(1,sum(trainCounts(1:(style-1)))), ones(1,noTrain), - ones(1,sum(trainCounts((style+1):end)))] ;
            
            % Prepare testing trainLabels
            testLabels =  [- ones(1,sum(testCounts(1:(style-1)))), ones(1,noTest), - ones(1,sum(testCounts((style+1):end)))] ;
            realClasses(sum(testCounts(1:(style-1)))+1:sum(testCounts(1:style))) = style;
            
            
            % --------------------------------------------------------------------
            % Stage B: Training a classifier
            % --------------------------------------------------------------------
            
            % Train the linear SVM
            lambda = 0.1 ; % Regularization parameter
            maxIter = 100000 ; % Maximum number of iterations
            [w, bias, info] = vl_svmtrain(trainDescriptors, trainLabels, lambda, 'MaxNumIterations', maxIter);
            %%%%%%%%%%%%%%%%%%%%%%%%%
            rows = size(cnnFeatures, 2);
            cols = size(cnnFeatures, 3);
            numPlanes = size(cnnFeatures, 1);
            testW = reshape(w, [numPlanes,rows,cols]);
            figure, imagesc(squeeze(sum(testW,1)));
            disp(styleNames(style));
             keyboard;

            %%%%%%%%%%%%%%%%%%%%%%%
            % Evaluate the scores on the training data
            trainScores = w' * trainDescriptors + bias ;
            allTrainScores(style, :) = trainScores - mean(trainScores);
            allTrainScores(style, :) = allTrainScores(style, :) / std(allTrainScores(style, :));
            
            
            % Visualize the precision-recall curve
            %             figure(2) ; clf ; set(2,'name','Precision-recall on train data') ;
            %             vl_pr(trainLabels, scores) ;
            
            % --------------------------------------------------------------------
            % Stage C: Classify the test images and assess the performance
            % --------------------------------------------------------------------
            
            % Test the linar SVM
            testScores = w' * testDescriptors + bias ;
            allTestScores(style, :) = testScores - mean(testScores);
            allTestScores(style, :) = allTestScores(style, :) / std(allTestScores(style, :));
            
            % Visualize the precision-recall curve
            %             figure(4) ; clf ; set(4,'name','Precision-recall on test data') ;
            %             vl_pr(testLabels, testScores) ;
            
            %         keyboard;
            
            % Print results
            [recall,precision,info] = vl_pr(testLabels, testScores) ;
            %     fprintf('Test AP: %.2f\n', info.auc) ;
            successRate = successRate + info.auc;
            
            %         [~,perm] = sort(testScores,'descend') ;
            %         fprintf('Correctly retrieved in the top %d: %d\n', noTest, sum(testLabels(perm(1:noTest)) > 0)) ;
        end
        fprintf('\nTest AP: %.4f\n', successRate/numClasses) ;
        
        
        
        [~, guesses] = max(allTestScores);
        myRate = sum(guesses==realClasses)/sum(testCounts);
        fprintf('\nMy AP: %.4f\n', myRate) ;
        grandTotal = grandTotal + myRate;
%         save([savePath '/' descriptor '_newsplit' num2str(split) '.mat'],'allTrainScores', 'allTestScores' );
    end
    
    fprintf('\nMean AP: %.4f\n', grandTotal/splits) ;
end
