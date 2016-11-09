close all; clear all; fclose all;clc;
infile = fopen('arcDataset/foldernames.txt','r');
folderList = cell(1,2);
count = 0;
while ~feof(infile)
    line = fgetl(infile);
    count = count + 1;
    path = ['arcDataset/images/' line '/*.*'];
    names = dir(path);
    num = numel(names);
    fprintf('%s\n',line);
    folderList{count,1} = line;
    folderList{count, 2} = num-2;
end
save('arcDataset/foldeList.mat','folderList');
fclose(infile);
