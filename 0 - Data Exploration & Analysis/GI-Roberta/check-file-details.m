% Load the original MRI image file
dataPath = '/Users/elizabethnemeti/Desktop/GI-Motility Project/GI-Motility-Data/4D-roberta-data/';
imageFile = 'FD_027_image.mat';
imageData = load(fullfile(dataPath, imageFile));

% Display details about the variables in the image data
disp('Details of the variables in the image file:');
whos('-file', fullfile(dataPath, imageFile));

% Load the mask file containing temporal data
maskFile = 'FD_027_temporal.mat';
maskData = load(fullfile(dataPath, maskFile));

% Display details about the variables in the mask data
disp('Details of the variables in the mask file:');
whos('-file', fullfile(dataPath, maskFile));
