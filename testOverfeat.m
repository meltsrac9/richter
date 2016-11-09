close all; clear all; fclose all;clc;
filters = {'*.jpg'; '*.png'; '*.gif'; '*.bmp'; '*.tif'};
[fileName, pathName, filterIndex] = uigetfile(filters,'Select the query image');
image = imread([pathName fileName]);
figure, imshow(image);
for level = 1:3:20
    tic;
    c = overfeat(image,level,'whole');
    fprintf('\nThe time taken for level %d is %f seconds.', level, toc);
    count = 0;
    if size(c,2)==1
        continue;
    end
    figure;
    if size(c,1)<=100
        step = 1;
    else
        step = floor(size(c,1)/100);
    end
    for x = 1:step:size(c,1);
        count = count+1;
        if count > 100
            break;
        end
        img = squeeze(c(x,:,:));
        subplot(10,10,count), imagesc(img);
        set(gca, 'XTick', []);
        set(gca, 'YTick', []);
        title(num2str(x));
    end
end


