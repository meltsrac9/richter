close all; clear all; clc;
folder1 = '/home/banerji/work/misc/Ray/arcDataset/data/cnn_whole_layer16';
folder2 = '/home/banerji/work/misc/Ray/arcDataset/data/cnn_whole_layer18';

load('arcDataset/folderList.mat');
for folder = 1: size(folderList, 1)
    path1 = [folder1 '/' folderList{folder, 1}];
    path2 = [folder2 '/' folderList{folder, 1}];
    names1 = dir([path1 '/*.mat']);
    names2 = dir([path2 '/*.mat']);
    count1 = numel(names1);
    count2 = numel(names2);
    if count1 == count2
        continue;
    end
    for file = 1:max([count1, count2])
        if strcmpi(names1(file).name, names2(file).name)
            continue;
        else
            fprintf('16: %s, 18: %s', [path1 '/' names1(file).name], [path2 '/' names2(file).name]);
            break;
        end
    end
end

        
        

