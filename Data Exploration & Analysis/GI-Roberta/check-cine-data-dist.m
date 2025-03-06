% % Data: Number of images/masks per slice (from your results)
% slices = 1:72;
% images_masks_count = [
%     0, 0, 0, 0, 0, 89, 132, 132, 132, 132, 264, 264, 264, 349, 396, 396, 396, ...
%     396, 396, 397, 514, 528, 528, 528, 564, 660, 660, 660, 660, 660, 660, 660, ...
%     660, 660, 660, 660, 660, 660, 660, 660, 660, 660, 660, 660, 660, 660, 660, ...
%     660, 660, 660, 660, 660, 659, 583, 536, 528, 528, 641, 426, 396, 396, 396, ...
%     396, 396, 353, 264, 150, 19, 0, 0, 0, 0];
% 
% % Create the bar plot
% figure;
% bar(slices, images_masks_count, 'FaceColor', [0.2, 0.6, 1], 'EdgeColor', 'none');
% xlabel('Slice Number');
% ylabel('Number of Images/Masks');
% title('Distribution of Images/Masks Across Slices');
% xticks(slices);
% xtickangle(90);  % Rotate x-axis labels for better visibility
% grid on;
% 
% % Adjust figure for better layout
% set(gca, 'FontSize', 12);
% xlim([1 72]);

% % Folder where your processed 3D data is saved
% output_folder = '/Users/elizabethnemeti/Desktop/all-cine-mri-pngs';
% 
% % Initialize a counter for each time point (1 to 132)
% time_points = 1:132;
% time_count = zeros(1, 132);
% 
% % Loop through all slice folders (cine-slice-1-data to cine-slice-72-data)
% for slice_num = 1:72
%     % Define the folder for the current slice
%     slice_folder = fullfile(output_folder, ['cine-slice-' num2str(slice_num) '-data']);
% 
%     % Get a list of all mask files in the folder (assuming mask files contain '_mask.png')
%     mask_files = dir(fullfile(slice_folder, '*_mask.png'));
% 
%     % Loop through each mask file and extract the time point from the filename
%     for k = 1:length(mask_files)
%         % Extract the filename without extension
%         [~, filename, ~] = fileparts(mask_files(k).name);
% 
%         % Find the time point in the filename (assuming format '_time_X_')
%         time_str = regexp(filename, '_time_(\d+)_', 'tokens');
%         if ~isempty(time_str)
%             time_idx = str2double(time_str{1}{1});
%             % Increment the counter for the corresponding time point
%             time_count(time_idx) = time_count(time_idx) + 1;
%         end
%     end
% end
% 
% % Create the bar plot for the distribution over time points
% figure;
% bar(time_points, time_count, 'FaceColor', [0.6, 0.2, 0.8], 'EdgeColor', 'none');
% xlabel('Time Point');
% ylabel('Number of Images/Masks');
% title('Distribution of Images/Masks Across Time Points');
% xticks(time_points);
% xtickangle(90);  % Rotate x-axis labels for better visibility
% grid on;
% 
% % Adjust figure for better layout
% set(gca, 'FontSize', 12);
% xlim([1 132]);


% Folder where your processed 3D data is saved
output_folder = '/Users/elizabethnemeti/Desktop/all-cine-mri-pngs';

% List of subjects
subjects = {'FD_027', 'FD_029', 'FD_030', 'FD_031', 'FD_032'};

% Initialize a counter to store the number of pairs per subject
pairs_per_subject = zeros(1, length(subjects));

% Loop through each subject
for s = 1:length(subjects)
    subject = subjects{s};
    
    % Initialize a counter for the number of pairs for the current subject
    total_pairs = 0;
    
    % Loop through all slice folders (cine-slice-1-data to cine-slice-72-data)
    for slice_num = 1:72
        % Define the folder for the current slice
        slice_folder = fullfile(output_folder, ['cine-slice-' num2str(slice_num) '-data']);
        
        % Get a list of all mask files for this subject in the folder
        mask_files = dir(fullfile(slice_folder, [subject '*_mask.png']));
        
        % Get a list of all image files for this subject in the folder
        image_files = dir(fullfile(slice_folder, [subject '*_image.png']));
        
        % The number of pairs is the minimum of image and mask files for consistency
        num_pairs = min(length(image_files), length(mask_files));
        
        % Add the number of pairs for this subject in the current slice to the total
        total_pairs = total_pairs + num_pairs;
    end
    
    % Store the total number of pairs for this subject
    pairs_per_subject(s) = total_pairs;
    
    % Print the number of pairs for this subject
    fprintf('Subject %s has %d pairs of images and masks.\n', subject, total_pairs);
end

% Create the bar plot for the number of pairs per subject
figure;
bar(categorical(subjects), pairs_per_subject, 'FaceColor', [0.4, 0.7, 0.9], 'EdgeColor', 'none');
xlabel('Subject');
ylabel('Number of Pairs (Images/Masks)');
title('Number of Pairs (Images/Masks) per Subject');
grid on;

% Adjust the layout
set(gca, 'FontSize', 12);
