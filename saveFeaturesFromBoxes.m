close all; clear all; fclose all;clc;


startImagePath = 'arcDataset/images';
startDataPath = 'arcDataset/data';
load('arcDataset/folderList.mat','folderList');
styleNames = folderList(:,1);
imageCounts = cell2mat(folderList(:,2));
clear folderList;
numBoxes = 10;
for style = 1:numel(styleNames)
    imagePath = [startImagePath '/' styleNames{style}];
    dataPath1 = [startDataPath '/patchAll/' styleNames{style} ];
    dataPath2 = [startDataPath '/patchImages/' styleNames{style} ];
    
    allNames = dir([imagePath '/*.*']);
    
    makeDir([dataPath1 '/']); %creates the data folder if it doesn't exist
    makeDir([dataPath2 '/']); %creates the data folder if it doesn't exist
    
    for x = 1:numel(allNames)
        if strcmp(allNames(x).name, '.') || strcmp(allNames(x).name, '..')
            continue;
        end
        inputFileName = [imagePath '/' allNames(x).name];
        
        fprintf('\n%s',inputFileName);
        inputImage = imread(inputFileName);
        boxes = findObjects4(inputImage, numBoxes);
        for box = 1:size(boxes, 1)
            fprintf('_%d_',box);
            blocks(box).cmin = boxes(box,1);
            blocks(box).rmin = boxes(box,2);
            blocks(box).cmax = boxes(box,3);
            blocks(box).rmax = boxes(box,4);
            boxImage = inputImage(blocks(box).rmin:blocks(box).rmax, blocks(box).cmin:blocks(box).cmax,:);
            temp = single(imresize(boxImage, [32, 32]));
            blocks(box).patchHOG = vl_hog(temp, 8);
            
            saveFile2 = [dataPath2 '/' allNames(x).name(1:end-4) '_' num2str(box) '.jpg'];
            imwrite(boxImage, saveFile2, 'jpg');
            fprintf('.');
            
            blocks(box).lbp = lbp(boxImage)';
            fprintf('.');
            blocks(box).tinyIm = tinyImage(boxImage);
            fprintf('.');
            blocks(box).colorHist = colorHistogram(boxImage);
            fprintf('.');
            blocks(box).colorLBP = colorlbp(boxImage)';
            fprintf('.');
            
        end
        
        saveFile1 = [dataPath1 '/' allNames(x).name(1:end-4) '.mat'];
        
        save(saveFile1,'boxes', 'blocks');
        clear boxes blocks;
        
        fprintf('\n');
        
        
    end
end
