Author: Elizabeth N.

% Path to the reconstructed .nii file
dataPath = '/Users/elizabethnemeti/Desktop/reconstructed-og/FD_027/';
reconstructedFile = 'FD_027_time_1_reconstructed.nii';

% Load the NIfTI file
niftiData = niftiread(fullfile(dataPath, reconstructedFile));
niftiInfo = niftiinfo(fullfile(dataPath, reconstructedFile));

% Extract the reconstructed volume
reconstructedVolume = double(niftiData); % Convert to double for processing

% Display the size of the reconstructed volume
disp('Reconstructed volume dimensions:');
disp(size(reconstructedVolume));

% Normalize the volume for visualization
volume_normalized = reconstructedVolume / max(reconstructedVolume(:));

% Create a 3D isosurface visualization
figure;
isosurface(volume_normalized, 0.5);  % 0.5 is the isosurface threshold
axis tight;
axis equal;
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');
title('3D Visualization of Reconstructed Volume');
colormap gray;
lighting gouraud;
camlight headlight;
