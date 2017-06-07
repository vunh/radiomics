function seg_map = dicomrt2matlab_v2(rtssfile, imagedir, segdir)

%% Parse input
if nargin < 2
  imagedir = '';
end
if nargin < 3
  segdir = '';
end

if isempty(imagedir)
  imagedir = fileparts(rtssfile);
end
if isempty(segdir)
  segdir = imagedir;
end




%% Load DICOM headers
fprintf('Reading image headers...\n');
rtssheader = dicominfo(rtssfile);
imageheaders = loadDicomImageInfo(imagedir, rtssheader.StudyInstanceUID);


%% Read contour sequences
fprintf('Converting RT structures...\n');
contours = readRTstructures(rtssheader, imageheaders); %#ok<NASGU>
%contours = convexPoints2bin(contours, imageheaders); %#ok<NASGU>


%% Save segmentations
if strcmp(imagedir, segdir)
  [~, name, ~] = fileparts(rtssfile);
else
  [~, name, ~] = fileparts(imagedir);
end
%files_out{1} = [segdir filesep name '.mat'];
%fprintf('Writing "%s"...\n', files_out{1});
%save(files_out{1}, 'contours', 'rtssheader', 'imageheaders', '-v7.3');
contours.Segmentation = permute(contours.Segmentation, [2, 1, 3]);
seg_map.contours = contours;
seg_map.rtssheader = rtssheader;
seg_map.imageheaders = imageheaders;


