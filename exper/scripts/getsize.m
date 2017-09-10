path_to_input = './normal';
path_to_output = '.';
fid = fopen('image_size.txt', 'wt');

iids = dir(fullfile(path_to_input, '*.jpg'));
for i = 1: length(iids)
    mask = imread(fullfile(path_to_input, iids(i).name));
    name = iids(i).name;
    [height, width, channel] = size(mask);
    fprintf( fid, '%s, %f, %f, %f\n', name, height, width, channel);
end