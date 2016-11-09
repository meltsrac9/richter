function [ output ] = overfeat( image, layer, mode )
%     if network == 0
%         path_to_weights = '/home/banerji/work/overfeat/data/default/net_weight_0';
%     else
%         path_to_weights = '/home/banerji/work/overfeat/data/default/net_weight_1';
%     end
    if nargin==1
        layer = 20;
        mode = 'whole';
    elseif nargin==2
        mode = 'whole';
    end
    if size(image, 3)==1
        image(:,:,2) = image;
        image(:,:,3) = image(:,:,1);
    end
    if strcmpi(mode,'whole')
        image = imresize(image, [231, 231]);
    elseif strcmpi(mode,'crop')
        smaller = min(size(image,1), size(image,2));
        image = image(1:smaller, 1:smaller, :);
        image = imresize(image, [231, 231]);
    else
        error('Invalid mode. Should be whole or crop.');
    end
    % Create a random filename
    alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_';
    count = 0;
    while(count==0 || exist(name, 'file'))
        letters = randperm(numel(alphabet), 8);
        name = ['temp/' alphabet(letters) '.jpg'];
        count = count + 1;
    end
    
    imwrite(image, name, 'jpg');
%     command = ['/home/banerji/work/overfeat/bin/linux_64/overfeatcmd ' path_to_weights ' -1 ' num2str(network) ' ' num2str(layer) ' ' 'temp/temp1.jpg'];
    command = ['/home/banerji/work/overfeat/bin/linux_64/overfeat -L ' num2str(layer) ' ' name];
%     fprintf('\n%s\n',command);
    [~, cmdout] = system(command);
%     fprintf('%s', cmdout);
    delete(name);
    c = strsplit(cmdout);
    d1 = str2num(c{1});
    d2 = str2num(c{2});
    d3 = str2num(c{3});
    output = zeros(d1, d2, d3(1), 'single');
    count = 1;
    for x = 1:d1
        for y = 1:d2
            for z = 1:d3(1)
                if numel(d3)==2 && count==1
                    output(x, y, z) = d3(2);
                else
                    output(x, y, z) = str2num(c{count + 2});
                end
                count = count + 1;
            end
        end
    end   
end

