close all; clear all; fclose all;clc;

startImagePath = 'arcDataset/images';
startDataPath = 'arcDataset/data';
load('arcDataset/folderList.mat','folderList');
count = 0;
numBoxes = 10;
mainLayers = [3, 6, 9, 12, 16, 18, 20];
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
        boxes = findObjects4(inputImage, numBoxes);

        count = count + 1;
        for layer = mainLayers
            dataPath = [startDataPath '/cnn_box_layer' num2str(layer) '/' folderList{style,1} ];
            makeDir([dataPath '/']); %creates the data folder if it doesn't exist
            saveFile = [dataPath '/' allNames(x).name(1:end-3) 'mat'];
            
            if(exist(saveFile,'file'))
                continue;
            end
            for box = 1:size(boxes, 1)
                blocks(box).cmin = boxes(box,1);
                blocks(box).rmin = boxes(box,2);
                blocks(box).cmax = boxes(box,3);
                blocks(box).rmax = boxes(box,4);
                boxImage = inputImage(blocks(box).rmin:blocks(box).rmax, blocks(box).cmin:blocks(box).cmax,:);
                blocks(box).cnnFeatures = overfeat(boxImage,layer,'whole');
            end
            save(saveFile,'blocks', 'boxes');
            clear blocks;
            fprintf('..%d',layer);
        end
        clear boxes;
             
    end
end
fprintf('\nDone!\n');

