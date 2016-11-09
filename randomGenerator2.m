close all; clear all; clc;

load('arcDataset/folderList.mat');
nums = cell2mat(folderList(:,2));
runs = 5;
numTrain = 30;
numTest = nums - numTrain;

for r = 1:runs
    testFileList = zeros(numel(nums), max(numTest));
    fprintf('Generating test split %d\n', r);
    for class = 1:numel(nums)
        testFileList(class,1:numTest(class)) = randperm(nums(class),numTest(class));
    end
    save(['arcDataset/testFileList_' num2str(r) '.mat'],'testFileList');
end
        
    

