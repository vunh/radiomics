% Using mask to extract tumor
function measure_tumor_range()

src_dir = '/nfs/bigbrain/vhnguyen/projects/radiomics/dataset/nsclc_60';
%des_dir = '/nfs/bigbrain/vhnguyen/projects/radiomics/dataset/nsclc_60_tumor';

files = dir(fullfile(src_dir, 'msk_*.mat'));
files = {files.name};

box = [];
for i = 1:length(files)
%for i = 11:11
	disp(files{i});
	msk = load(fullfile(src_dir, files{i}));
	msk = msk.segmentation;

	stats = regionprops(msk, 'BoundingBox');
	stats_size = size(stats);
	if (stats_size(1) > 0)
		top_left = stats(1).BoundingBox(1:3);
		bottom_right = top_left + stats(1).BoundingBox(4:6);
		top_left = floor(top_left);
		bottom_right = ceil(bottom_right);

		box = [box; [bottom_right - top_left + 1]];
	end
end

max_height = max(box(:,1));
min_height = min(box(:,1));
max_width = max(box(:,2));
min_width = min(box(:,2));
max_depth = max(box(:,3));
min_depth = min(box(:,3));

disp(sprintf('max_height %d, min_height %d, max_width %d, min_width %d, max_depth %d, min_depth %d', ceil(max_height), ceil(min_height), ceil(max_width), ceil(min_width), ceil(max_depth), ceil(min_depth)));

end
