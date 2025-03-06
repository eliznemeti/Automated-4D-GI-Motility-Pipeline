% Define the folder where your raw data is located
input_folder = '/Users/elizabethnemeti/Desktop/roberta-data';

% Folder where your processed 3D data will be saved
output_folder = '/Users/elizabethnemeti/Desktop/all-cine-mri-pngs';

% Make sure the output folder exists (create it if it doesn't)
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Create subfolders for each slice
for slice_num = 1:72
    slice_folder = fullfile(output_folder, ['cine-slice-' num2str(slice_num) '-data']);
    if ~exist(slice_folder, 'dir')
        mkdir(slice_folder);
    end
end

% List of subject files (assuming image/mask pairs for each subject)
subjects = {'FD_027', 'FD_029', 'FD_030', 'FD_031', 'FD_032'};

% Loop through each subject and process their data
for i = 1:length(subjects)
    subject = subjects{i};
    
    % Load the image and mask files
    image_file = fullfile(input_folder, [subject '_image.mat']);
    mask_file = fullfile(input_folder, [subject '_temporal.mat']);
    
    % Load the data from the .mat files
    image_data = load(image_file);
    mask_data = load(mask_file);
    
    % Use the correct variable names from the .mat files
    image_vol = image_data.data;         % 4D image volume (X,Y,Z,Time)
    mask_vol = mask_data.BWprop_LRT;     % 4D mask volume (X,Y,Z,Time)

    % Loop through each slice (Z direction)
    for slice_num = 1:size(image_vol, 3)
        
        % Get the folder for this slice
        slice_folder = fullfile(output_folder, ['cine-slice-' num2str(slice_num) '-data']);
        
        % Loop through time points in the image and mask volumes (4th dimension is time)
        for t = 1:size(image_vol, 4)
            % Extract 2D image slice for the current time point and slice
            image_slice = squeeze(image_vol(:, :, slice_num, t));
            
            % Extract the corresponding mask slice for the current time point
            mask_slice = squeeze(mask_vol(:, :, slice_num, t));
            
            % Resize the image and mask to 256x256
            image_resized = imresize(image_slice, [256, 256]);
            mask_resized = imresize(mask_slice, [256, 256]);

            % Normalize mask values to 0-1 (binary mask) if needed
            mask_resized = mask_resized > 0;  % Convert to binary mask (0 or 1)

            % Create filenames for saving the PNG files
            image_filename = fullfile(slice_folder, [subject '_time_' num2str(t) '_slice_' num2str(slice_num) '_image.png']);
            mask_filename = fullfile(slice_folder, [subject '_time_' num2str(t) '_slice_' num2str(slice_num) '_mask.png']);
            
            % Save the image and mask as PNGs
            imwrite(uint8(image_resized), image_filename);  % Ensure image is saved as uint8
            imwrite(uint8(mask_resized * 255), mask_filename);  % Save mask as binary PNG (0 or 255)
        end
    end
end
