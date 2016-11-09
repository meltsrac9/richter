close all; clear all; fclose all;clc;
% for layer = 1:20
%     disp(layer);
%     startImagePath = ['arcDataset/data/cnn_whole_layer' num2str(layer)];
    startImagePath = 'arcDataset/data/colorHist';

    load('arcDataset/folderList.mat','folderList');
    
    for style = 1:size(folderList,1)
        imagePath = [startImagePath '/' folderList{style,1}];
        
        allNames = dir([imagePath '/*.*']);
        
        
        for x = 1:numel(allNames)
            if strcmp(allNames(x).name, '.') || strcmp(allNames(x).name, '..')
                continue;
            end
            
            inputFileName = [imagePath '/' allNames(x).name];
            if size(allNames(x).name,2) >= 100
                newFileName = [imagePath '/' allNames(x).name(1:95) allNames(x).name(end-3:end)];
                fprintf('\n%s',inputFileName);
                
                movefile(inputFileName,newFileName) ;
            end
        end
    end
% end
fprintf('\nDone!\n');

