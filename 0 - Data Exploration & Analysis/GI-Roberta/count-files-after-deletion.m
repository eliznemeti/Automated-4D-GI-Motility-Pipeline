% Folder where your processed 3D data is saved
output_folder = '/Users/elizabethnemeti/Desktop/all-cine-mri-pngs';

% Initialize counters
total_images = 0;
total_masks = 0;

% Loop through all slice folders (cine-slice-1-data to cine-slice-72-data)
for slice_num = 1:72
    % Define the folder for the current slice
    slice_folder = fullfile(output_folder, ['cine-slice-' num2str(slice_num) '-data']);
    
    % Get a list of all image and mask files in the folder
    image_files = dir(fullfile(slice_folder, '*_image.png'));
    mask_files = dir(fullfile(slice_folder, '*_mask.png'));
    
    % Count the number of files in this slice folder
    num_images = length(image_files);
    num_masks = length(mask_files);
    
    % Add to the total count
    total_images = total_images + num_images;
    total_masks = total_masks + num_masks;
    
    % Print the count for the current slice folder
    fprintf('Slice %d: %d images, %d masks\n', slice_num, num_images, num_masks);
end

% Print the total number of files across all slice folders
fprintf('\nTotal remaining images: %d\n', total_images);
fprintf('Total remaining masks: %d\n', total_masks);
