% Param
slice_id = 43;

x=load('a/1/000000.mat');
[X, map] = dicomread(x.imageheaders{1,slice_id}.Filename);
seg_mesh = x.contours.Segmentation;
mask = seg_mesh( : , : , slice_id)';
figure(1); imagesc(X);
X(mask) = 3000;
figure(2); imagesc(X);

