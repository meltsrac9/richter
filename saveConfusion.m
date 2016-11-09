close all; clear all; clc;
path = 'arcDataset/results/cnn_whole_layer16_svmScores';
load([ path '/newConfusion.mat']);
saveFileName = [path '/newConfusion.jpg'];

confAverage = mean(conf,3);
testCounts = sum(conf(:,:,1),2);
confFinal = confAverage ./ repmat(testCounts, 1, size(conf,2));
A = figure, imagesc(confFinal);
saveas(A, saveFileName, 'jpeg');