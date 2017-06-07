function convert_dicom()

src_dir = '/nfs/bigbrain/vhnguyen/projects/radiomics/nsclc/NSCLC-Radiomics';
des_dir = '/nfs/bigbrain/vhnguyen/projects/radiomics/dataset/nsclc_60';


dicom_fold = list_bare_folders(src_dir);
%parpool(8);
dict = cell(length(dicom_fold), 1);
no_sample = length(dicom_fold);
for i_dicom = 1:no_sample
    dicom_name = dicom_fold{i_dicom};
    
    
    inter = list_bare_folders(fullfile(src_dir, dicom_name));
    host_fold = fullfile(src_dir, dicom_name, inter{1});
    sub_folds = list_bare_folders(host_fold);
	if (length(sub_folds) < 2)
			continue;
	end

    sub_fold_file_1 = fullfile(host_fold, sub_folds{1});
    sub_fold_file_2 = fullfile(host_fold, sub_folds{2});
    
    img_fold = sub_fold_file_1;
    msk_fold = sub_fold_file_2;
    if (length(list_bare_files(sub_fold_file_1)) == 1)
        img_fold = sub_fold_file_2;
        msk_fold = sub_fold_file_1;
    end
    
    list_msk_file = list_bare_files(msk_fold);
    msk_file = fullfile(msk_fold, list_msk_file{1});
    hdr = dicominfo(fullfile(msk_file));
    if strcmp(hdr.Modality, 'RTSTRUCT')
        fprintf('Converting %s\n', dicom_name);
        %files_out = [files_out dicomrt2matlab(fullfile(directory, files(f).name))];
        seg_map = dicomrt2matlab_v2(msk_file, img_fold);
        
        % Convert and Save image block
        saved_img_block = convertImg(img_fold, seg_map.imageheaders);
        
        img = saved_img_block;
        segmentation = seg_map.contours.Segmentation;
        
        save(fullfile(des_dir, ['img_' num2str(i_dicom) '.mat']), 'img');
        save(fullfile(des_dir, ['msk_' num2str(i_dicom) '.mat']), 'segmentation');
        dict{i_dicom} = [dicom_name ' ' num2str(i_dicom)];
        
    end
end

%delete(gcp);

fid=fopen(fullfile(des_dir, 'info.txt'),'w');
for i = 1:length(dict)
		if (length(dict{i}) == 0)
				continue;
		end
		fprintf(fid, '%s\n', dict{i});
end
fclose(fid);


end

function saved_mat = convertImg(img_dir, imageheaders)

dicom_list = list_bare_files(img_dir);

[X, ~] = dicomread(imageheaders{1,1}.Filename);
saved_mat = zeros(size(X, 1), size(X, 2), length(dicom_list));
for i = 1:length(dicom_list)
    [X, ~] = dicomread(imageheaders{1,i}.Filename);
    saved_mat(:, :, i) = X;
end

end


function list = list_bare_folders (parent)

list = {};

files = dir(parent);
for f = 1:length(files)
    if (strcmp(files(f).name(1), '.'))
        continue;
    end
    
    if (~files(f).isdir)
        continue;
    end
    
    list = [list {files(f).name}];
end

end

function list = list_bare_files (parent)

list = {};

files = dir(parent);
for f = 1:length(files)
    if (strcmp(files(f).name(1), '.'))
        continue;
    end
    
    if (files(f).isdir)
        continue;
    end
    
    list = [list {files(f).name}];
end

end
