close all; clear all; fclose all;clc;

startImagePath = 'arcDataset/images';
startDataPath = 'arcDataset/data';
load('arcDataset/folderList.mat','folderList');
count = 0;

for style = 1:size(folderList,1)
    imagePath = [startImagePath '/' folderList{style,1}];
    
    allNames = dir([imagePath '/*.*']);
    for x = 1:numel(allNames)
        if strcmp(allNames(x).name, '.') || strcmp(allNames(x).name, '..')
            continue;
        end
        inputFileName = [imagePath '/' allNames(x).name];
        
        fprintf('\n%s',inputFileName);
        inputImage = imread(inputFileName);
        count = count + 1;
        for layer = 1:1:20
            dataPath = [startDataPath '/cnn_whole_layer' num2str(layer) '/' folderList{style,1} ];
            makeDir([dataPath '/']); %creates the data folder if it doesn't exist
            saveFile = [dataPath '/' allNames(x).name(1:end-3) 'mat'];
            
            if(exist(saveFile,'file'))
                continue;
            end
            
            cnnFeatures = overfeat(inputImage,layer,'whole');
            save(saveFile,'cnnFeatures');
            fprintf('..%d',layer);

        end
             
    end
end
fprintf('\nDone!\n');

