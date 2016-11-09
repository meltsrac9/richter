function [ output] = colorHistogram( image )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
if size(image, 3)==1
    image(:,:,2) = image;
    image(:,:,3) = image(:,:,2);
end
output = [imhist(image(:,:,1)); imhist(image(:,:,2)); imhist(image(:,:,3))]; 
output = output / (size(image,1) * size(image,2));
end

