close all; clearvars;
% read in 4D NIfTI image file 
img_antrum_seg = niftiread('/Users/elizabethnemeti/Desktop/Subject_reconstruction_comparisons/FD_027_4D_reconstructed_original.nii');
[~, ~, ~, nt] = size(img_antrum_seg); % 4th dimension ntrepresents number of time points
disp('Loaded 4D data.');

%% 1. Automate identifying the antrum region
% We precompute the ROI for the antrum based on the anatomy at the first time point
A = img_antrum_seg(:,:,:,1); % for first time point
A = smooth3(A / max(A(:)), 'box', 5); % smooth and normalize
stats = regionprops3(A, 'BoundingBox', 'Centroid'); % regionprops3 helps get the spatial info
centroid = stats.Centroid;
bounding_box = stats.BoundingBox;

% Estimate the antrum ROI based on lower portion of the bounding box
[z_dim] = size(A, 3); % get total depth (z-dimension)
antrum_start = round(0.7 * z_dim); % using a general rule here that the antrum starts at 70% depth, and then using this point and down for the ROI as an approximation
antrum_cubepos = [1, 1, antrum_start, size(A, 1), size(A, 2), size(A, 3) - antrum_start + 1]; % here is where cropping bounds are defined - how far the cube will span
disp(['Computed antrum ROI: ', num2str(antrum_cubepos)]);

%% 2. Rotate and Crop Antrum
low_antrum_vol_4D = [];

% for loop to process each 3d volume for every time point 
for t = 1:nt
    B = img_antrum_seg(:,:,:,t); % time point currently
    B = smooth3(B / max(B(:)), 'box', 5); % smooth and normalize
    B = imtranslate(B, -centroid, 'OutputView', 'full'); % align antrum centroid with center via translation
    
    % apply 3D rotations along the specified axes
    angles = [0, 0, 0]; % HERE is where you must define fixed angles if you'd like to rotate
    for d = 1:3
        if angles(d) ~= 0
            axis = zeros(1, 3); axis(d) = 1; % rotation axis defined here
            B = imrotate3(B, angles(d), axis, 'crop');
        end
    end

    % Crop the antrum region
    low_antrum_vol = imcrop3(B, antrum_cubepos);

    % speed up processing by preallocating memory instead of re-resizing
   
    if t == 1
        [x, y, z] = size(low_antrum_vol);
        low_antrum_vol_4D = zeros(x, y, z, nt);
    end
    
    % Store cropped volume
    [x, y, z] = size(low_antrum_vol);
    if x <= size(low_antrum_vol_4D, 1) && y <= size(low_antrum_vol_4D, 2) && z <= size(low_antrum_vol_4D, 3)
        low_antrum_vol_4D(1:x, 1:y, 1:z, t) = low_antrum_vol;
    else
        error('Cropped volume dimensions do not match preallocated size for time point %d.', t);
    end
end
disp('Completed rotation and cropping for all time points.');

%% 3. Visualize Cropped Volume (First Time Point)
figure('Color', 'k');
volshow(low_antrum_vol_4D(:,:,:,1));