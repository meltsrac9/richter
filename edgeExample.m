close all; clear all;
img = imread('/home/banerji/Pictures/2February.jpg');
img = imresize(img, [500 NaN]);
img = rgb2gray(img);
figure, imshow(img);
li = img(100, :);
figure, plot(li);
img2 = img(:,2:end);
img2(:,750) = 0;
figure, imshow(img2);
vert = abs(img - img2);
vert  = normalize(vert);
figure, imshow(vert);
img2 = img(2:end,:);
img2(500,:) = 0;
figure, imshow(img2);
horz = abs(img - img2);
horz  = normalize(horz);
figure, imshow(horz);
all = horz+vert;
all= normalize(all);
figure, imshow(all);

figure, imshow(edge(img, 'roberts'));
figure, imshow(edge(img, 'sobel'));
figure, imshow(edge(img, 'prewitt'));
figure, imshow(edge(img, 'canny'));