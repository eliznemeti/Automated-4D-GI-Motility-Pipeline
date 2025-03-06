% MATLAB Script to Resize All Images in a Directory to 256x256 Pixels

% ==================== Step 1: Specify the Directory Path ====================
% Define the path to the images directory
imagesDir = '/Users/elizabethnemeti/Desktop/images-temporal-roberta';

% Verify that the directory exists
if ~isfolder(imagesDir)
    error('The specified directory does not exist: %s', imagesDir);
end

% ==================== Step 2: List All Image Files ============================
% Define the list of supported image file extensions
imageExtensions = {'*.png', '*.jpg', '*.jpeg', '*.bmp'};

% Initialize an empty struct array to store image file information
imageFiles = struct([]);

% Iterate over each extension and gather all matching files
for i = 1:length(imageExtensions)
    files = dir(fullfile(imagesDir, imageExtensions{i}));
    imageFiles = [imageFiles; files]; %#ok<AGROW>
end

% Check if any image files were found
if isempty(imageFiles)
    error('No image files found in the specified directory: %s', imagesDir);
end

% ==================== Step 3: Create an Output Directory =======================
% Define the output directory for resized images
outputDir = fullfile(imagesDir, 'resized_images');

% Check if the output directory exists; if not, create it
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
    fprintf('Created output directory: %s\n', outputDir);
else
    fprintf('Output directory already exists: %s\n', outputDir);
end

% ==================== Step 4: Resize Images and Save Them ========================
% Define the target size
targetSize = [256, 256]; % [Height, Width]

% Iterate over each image file
for idx = 1:length(imageFiles)
    % Get the current image file name
    currentFile = imageFiles(idx).name;
    
    % Construct the full path to the current image
    currentPath = fullfile(imagesDir, currentFile);
    
    % Read the image
    img = imread(currentPath);
    
    % Determine if the image is grayscale or RGB
    if size(img, 3) == 3
        % If RGB, convert to grayscale
        img = rgb2gray(img);
    end
    
    % Resize the image to 256x256
    resizedImg = imresize(img, targetSize);
    
    % Construct the output file path
    outputFilePath = fullfile(outputDir, currentFile);
    
    % Save the resized image
    imwrite(resizedImg, outputFilePath);
    
    % Get the size of the resized image
    [height, width] = size(resizedImg);
    
    % Print the final dimensions
    fprintf('Resized Image: %s - Dimensions: %dx%d\n', currentFile, height, width);
end

% ==================== Completion Message ======================================
fprintf('All images have been resized and saved to: %s\n', outputDir);
