% Param


x=load('NSCLC/des/msk_1.mat');
X = load('NSCLC/des/img_1.mat');
X = X.img;
seg_mesh = x.segmentation;

disp (size(X,3));
for slice_id = 40:size(X, 3)
    disp(slice_id);
    mask = seg_mesh( : , : , slice_id);
    slice = X(:, :, slice_id);
    figure(1); imagesc(slice);
    slice(mask) = 3000;
    figure(2); imagesc(slice);
    pause;
end





