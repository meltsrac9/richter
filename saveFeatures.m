close all; clear all; fclose all;clc;

startImagePath = 'arcDataset/images';
startDataPath = 'arcDataset/data';
load('arcDataset/folderList.mat','folderList');

for style = 1:size(folderList,1)
    imagePath = [startImagePath '/' folderList{style,1}];
    dataPath1 = [startDataPath '/lbp/' folderList{style,1} ];
    dataPath2 = [startDataPath '/colorHist/' folderList{style,1} ];
    dataPath3 = [startDataPath '/tinyIm/' folderList{style,1}];
    
    allNames = dir([imagePath '/*.*']);
    
    makeDir([dataPath1 '/']); %creates the data folder if it doesn't exist
    makeDir([dataPath2 '/']); %creates the data folder if it doesn't exist
    makeDir([dataPath3 '/']); %creates the data folder if it doesn't exist

    for x = 1:numel(allNames)
        if strcmp(allNames(x).name, '.') || strcmp(allNames(x).name, '..') 
            continue;
        end
        saveFile1 = [dataPath1 '/' allNames(x).name(1:end-3) 'mat'];
        saveFile2 = [dataPath2 '/' allNames(x).name(1:end-3) 'mat'];
        saveFile3 = [dataPath3 '/' allNames(x).name(1:end-3) 'mat'];

        inputFileName = [imagePath '/' allNames(x).name];
        
        fprintf('\n%s',inputFileName);
        inputImage = imread(inputFileName);
                    
%         LBP = lbp(inputImage);
%         save(saveFile1,'LBP');
%         
%         colorHist = colorHistogram(inputImage);
%         save(saveFile2,'colorHist');
        
        tinyIm = tinyImage(inputImage);
        save(saveFile3,'tinyIm');
    end
end
fprintf('\nDone!\n');

