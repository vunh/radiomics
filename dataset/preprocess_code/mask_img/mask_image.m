% Using mask to extract tumor
function mask_image()

src_dir = '/nfs/bigbrain/vhnguyen/projects/radiomics/dataset/nsclc_60';
des_dir = '/nfs/bigbrain/vhnguyen/projects/radiomics/dataset/nsclc_60_tumor';

max_height = 119;
max_width = 148;
max_depth = 48;

max_dimension = [117, 146];
pad_dimension = [0, 0];
max_dimension = max_dimension + pad_dimension;
standard_dimension = [128, 128, 64];

scale_arr = [standard_dimension(1:2) ./ max_dimension]
scale = min(scale_arr);
disp(scale);

files = dir(fullfile(src_dir, 'img_*.mat'));
files = {files.name};

for i = 1:length(files)
	disp(i);
	disp(files{i});
	img_name = files{i};
	img = load(fullfile(src_dir, img_name));
	msk = load(fullfile(src_dir, ['msk' img_name(4:end)]));
	
	img = img.img;
	msk = msk.segmentation;
	neg_msk = ~msk;

	img(neg_msk) = 0;

	stats = regionprops(msk, 'BoundingBox');
	stats_size = size(stats);
	if (stats_size(1) == 0)
		continue;
	end
	
	% Crop tumor
	top_left = stats(1).BoundingBox(1:3);
	bottom_right = top_left + stats(1).BoundingBox(4:6);
	
	top_left = floor(top_left);
	bottom_right = ceil(bottom_right);

	org_img_size = size(img);
	top_left = max([1 1 1], top_left);
	bottom_right = min(org_img_size, bottom_right);

	org_tumor = img(top_left(1):bottom_right(1), top_left(2):bottom_right(2), top_left(3):bottom_right(3));

	% Resize
	org_tumor_size = size(org_tumor);
	resized_tumor = resize_3d(org_tumor, scale, standard_dimension);

	% Relocate in standard volume
	resized_tumor_size = size(resized_tumor);
	disp([org_tumor_size; resized_tumor_size]);
	new_img = zeros(standard_dimension);
	new_loc = ceil((standard_dimension - resized_tumor_size + 1) / 2);
	new_img(new_loc(1):new_loc(1)+resized_tumor_size(1)-1, new_loc(2):new_loc(2)+resized_tumor_size(2)-1, new_loc(3):new_loc(3)+resized_tumor_size(3)-1) = resized_tumor;

	norm_tumor = new_img;

	save(fullfile(des_dir, img_name), 'norm_tumor');
end

end


% this function actually resize x and y dimension of the volume
% standard_dimension is the boundary for scaling
function res = resize_3d(img, scale, standard_dimension)
depth = size(img, 3);

img_size = size(img);
img_new_size = floor([img_size(1:2)*scale img(3)]);
img_new_size = min(img_new_size, standard_dimension);
res = zeros(img_new_size);
for i = 1:depth
	slice = imresize(img(:,:,i), img_new_size(1:2));
	res(:,:,i) = slice;
end


end


