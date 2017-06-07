function files_out = scanDir(directory)

files_out = {};

files = dir(directory);
for f = 1:length(files)
%parfor f = 1:length(files)
  if files(f).isdir
    if ~strcmp(files(f).name(1), '.')
      files_out = [files_out scanDir(fullfile(directory, files(f).name))];
    end
  else
    try
      hdr = dicominfo(fullfile(directory, files(f).name));
      if strcmp(hdr.Modality, 'RTSTRUCT')
        fprintf('Converting %s\n', fullfile(directory, files(f).name));
        %files_out = [files_out dicomrt2matlab(fullfile(directory, files(f).name))];
        files_out = [files_out dicomrt2matlab(fullfile(directory, files(f).name), 'a/1')];
      end
    catch ME
      
    end
  end
end
