close all; clear all;
im1 = imread('/home/banerji/Desktop/colorful-clothing.jpg');
im2 = imread('/home/banerji/Desktop/low-contrast.jpg');
im2 = rgb2gray(im2);
figure, imshow(im1);
figure, imshow(im2);
h = imhist(im2);
figure;bar(h)
im2 = normalize(im2);
figure, imshow(im2);
h = imhist(im2);
figure;bar(h)
i = [imhist(im1(:,:,1));imhist(im1(:,:,2));imhist(im1(:,:,3))];
figure;bar(i)


