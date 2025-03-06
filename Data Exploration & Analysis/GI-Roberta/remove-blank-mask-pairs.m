% Folder where your processed 3D data is saved
output_folder = '/Users/elizabethnemeti/Desktop/all-cine-mri-pngs';

% Create a list to store filenames of deleted files
deleted_files = {};

% Loop through all slice folders (cine-slice-1-data to cine-slice-72-data)
for slice_num = 1:72
    % Define the folder for the current slice
    slice_folder = fullfile(output_folder, ['cine-slice-' num2str(slice_num) '-data']);
    
    % Get a list of all mask files in the folder (assuming mask files contain '_mask.png')
    mask_files = dir(fullfile(slice_folder, '*_mask.png'));
    
    % Loop through each mask file
    for k = 1:length(mask_files)
        % Get the full path of the mask file
        mask_filename = fullfile(mask_files(k).folder, mask_files(k).name);
        
        % Load the mask image
        mask = imread(mask_filename);
        
        % Check if the mask is all black
        if all(mask(:) == 0)
            % If the mask is all black, delete it and the corresponding image
            
            % Create the corresponding image filename
            image_filename = strrep(mask_filename, '_mask.png', '_image.png');
            
            % Delete the mask and image files
            delete(mask_filename);
            delete(image_filename);
            
            % Add the filenames to the list of deleted files
            deleted_files{end+1} = mask_filename; %#ok<SAGROW>
            deleted_files{end+1} = image_filename; %#ok<SAGROW>
            
            % Print message indicating which files were deleted
            fprintf('Deleted: %s and %s\n', mask_filename, image_filename);
        end
    end
end

% At the end, print the total number of deleted files
fprintf('\nTotal deleted files: %d\n', length(deleted_files));

% Optionally, write the list of deleted files to a text file
deleted_files_list = fullfile(output_folder, 'deleted_files_list.txt');
fid = fopen(deleted_files_list, 'w');
for i = 1:length(deleted_files)
    fprintf(fid, '%s\n', deleted_files{i});
end
fclose(fid);

fprintf('Deleted file list saved to %s\n', deleted_files_list);
