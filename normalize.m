function [ output ] = normalize( input )

input = input - min(min(input));
input = double(input) * 255 / double(max(max(input)));
output = uint8(input);

end

