clc;
path_to_input = '/media/Disk/wangfuyu/Carvana/test_hq/JPEGImages_large';
path_to_output = '/media/Disk/wangfuyu/Carvana/crop/test_crop/JPEGImages/';
mkdir(path_to_output);

list = 'test_list.txt';

max_w = 0
max_h = 0

for i = 1:5088,
    [name, x1, y1, x2, y2] = textread(list, '%s %f %f %f %f');
    img_name = fullfile(path_to_input, name{i,1});
    %img_name = [img_name(1:end-4) , '.png']
    display(img_name)
    im = imread(img_name);
    width = x2(i) - x1(i);
    height = y2(i) - y1(i);
    max_w = max(width, max_w);
    max_h = max(height, max_h);
    croped_im = imcrop(im, [x1(i), y1(i), width, height]);
    croped_name = fullfile(path_to_output, name{i,1});
    %croped_name = [croped_name(1:end-4) , '.png']
    imwrite(croped_im, croped_name);
end

display(max_w);
display(max_h);
