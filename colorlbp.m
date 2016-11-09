function h = colorlbp(image)
if size(image,3)==1
        image(:,:,2) = image;
        image(:,:,3) = image(:,:,2);
end
h = [lbp(image(:,:,1)), lbp(image(:,:,2)), lbp(image(:,:,3))];
