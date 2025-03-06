%% Initialize
close all; clearvars;
%load('TS_Displacement.mat');
%load('BW_temporalChecked.mat');
img_antrum_seg = niftiread('/Users/elizabethnemeti/Desktop/Subject_reconstruction_comparisons/FD_027_4D_reconstructed_original.nii'); % Path to the 4D volume

[~, ~, ~, nt] = size(img_antrum_seg);
disp('Loaded 4D data.');

%% Translate and Visualize Mesh (First Time Point)
A = img_antrum_seg(:,:,:,1); % First time point for alignment
A = smooth3(A / max(A(:)), 'box', 5); % Smooth and normalize
stats = regionprops3(A, 'Centroid');
centroid = stats.Centroid;
B = imtranslate(A, -centroid, 'OutputView', 'full'); % Translate to center
disp(['Non-zero elements after alignment: ', num2str(nnz(B))]);

% Visualize initial mesh
[faces, verts] = isosurface(B, 0.5);
figure, h = patch('Vertices', verts, 'Faces', faces, ...
    'FaceColor', [1 .65 .65], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
camlight;
title('Translated Stomach Mesh (First Time Point)');

%% Interactive Rotation
rotate_flag = 1;
final_angles = [0, 0, 0]; % Initialize rotation angles

while rotate_flag
    % Prompt user for rotation parameters
    prompt = {'Enter angle (+ counterclockwise):', 'Enter rotation axis ([x y z]):'};
    dlgtitle = '3D Rotation Parameters';
    dims = [1 35];
    definput = {'0', '[1 0 0]'}; % Default values
    opts.WindowStyle = 'normal';
    params = inputdlg(prompt, dlgtitle, dims, definput, opts);

    % Check if dialog was canceled
    if isempty(params)
        disp('Rotation dialog canceled. Exiting rotation process.');
        break;
    end

    % Parse user input
    angle = str2double(params{1});
    axis = str2num(params{2}); %#ok<ST2NM>

    % Validate parsed input
    if isnan(angle) || isempty(axis) || numel(axis) ~= 3
        warning('Invalid rotation parameters entered. Try again.');
        continue; % Restart the loop
    end

    % Update rotation
    C = imrotate3(B, angle, axis, 'crop');
    [faces, verts] = isosurface(C, 0.5);
    h.Vertices = verts;
    h.Faces = faces;
    drawnow;

    % Save final angles
    if isequal(axis, [1 0 0])
        final_angles(1) = final_angles(1) + angle;
    elseif isequal(axis, [0 1 0])
        final_angles(2) = final_angles(2) + angle;
    elseif isequal(axis, [0 0 1])
        final_angles(3) = final_angles(3) + angle;
    else
        warning('Invalid rotation axis. Skipping this rotation.');
    end

    % End rotation prompt
    end_rotation = questdlg('End rotation process?', 'Rotation', 'Yes', 'No', 'Yes');
    if strcmp(end_rotation, 'Yes')
        rotate_flag = 0;
    end
end

disp(['Final rotation angles: ', num2str(final_angles)]);

%% Rotate the 4D Data
BW_rot = zeros(size(img_antrum_seg)); % Preallocate rotated data
for t = 1:nt
    B = img_antrum_seg(:,:,:,t); % Current time point
    B = smooth3(B / max(B(:)), 'box', 5); % Smooth and normalize
    B = imtranslate(B, -centroid, 'OutputView', 'full'); % Translate to center
    
    % Apply rotation
    for d = 1:3
        if final_angles(d) ~= 0
            axis = zeros(1, 3); axis(d) = 1; % Define rotation axis
            B = imrotate3(B, final_angles(d), axis, 'crop');
        end
    end
    
    % Resize to match the original dimensions
    B_resized = imresize3(B, size(img_antrum_seg(:,:,:,1))); % Resize to match original size
    BW_rot(:,:,:,t) = B_resized; % Assign resized volume
end
disp('Rotated the 4D data.');

% Debug: Check non-zero elements in rotated data
disp(['Non-zero elements in rotated data (Time Point 1): ', num2str(nnz(BW_rot(:,:,:,1)))]);

%% Define and Crop ROI
figure, h2 = patch(isosurface(BW_rot(:,:,:,1), 0.5), ...
    'FaceColor', [1 .65 .65], 'EdgeColor', 'none');
camlight;
hold on;
voi = drawcuboid; % Define ROI interactively
cubepos = [floor(voi.Position(1:3)), ceil(voi.Position(4:6))];
cubepos(cubepos <= 0) = 1; % Ensure valid indices
disp(['Defined cuboid ROI: ', num2str(cubepos)]);

% Crop the volume
low_antrum_vol = imcrop3(BW_rot, cubepos);
disp(['Non-zero elements in cropped volume (Time Point 1): ', num2str(nnz(low_antrum_vol(:,:,:,1)))]);

%% Visualize Cropped Volume
figure('Color', 'k'); % Create a figure with a black background
volshow(low_antrum_vol(:,:,:,1)); % Display the volume

%% Calculate Profiles
profile3D_low_antrum_vol = squeeze(sum(low_antrum_vol, 2)); % Sum along y-axis
figure('Color', 'k'); % Create a figure with a black background
volshow(profile3D_low_antrum_vol); % Display the volume

profile_low_antrum_vol = squeeze(sum(low_antrum_vol, 3)); % Sum along z-axis
profile2D_low_antrum_vol = squeeze(sum(profile_low_antrum_vol, 1)); % Final 2D profile
figure;
imagesc(profile2D_low_antrum_vol);
colormap('hot');
colorbar;
title('2D Profile of Cropped Volume');
