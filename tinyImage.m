function tinyIM = tinyImage(image, side)
    if nargin==1
        side = 16;
    end
    if size(image, 3) == 1
        image(:,:,2) = image;
        image(:,:,3) = image(:,:,2);
    end
    im2 = imresize(image, [side side]);
    tinyIM = im2(:);