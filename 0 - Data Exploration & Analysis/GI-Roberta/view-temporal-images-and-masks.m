dataPath = '/Users/elizabethnemeti/Desktop/GI-Motility Project/GI-Motility-Data/4D-roberta-data';

% Pick subject files
imageFile = 'FD_027_image.mat';
maskFile = 'FD_027_temporal.mat';

% Load image and mask data
imageData = load(fullfile(dataPath, imageFile));
maskData = load(fullfile(dataPath, maskFile));

% Extract image and mask variables
data = imageData.data;        % Image data (192x156x72x132)
BWprop_LRT = maskData.BWprop_LRT;  % Mask data (192x156x72x132)

% Pick slice of interest to visualize
sliceOfInterest = 10; 

gifFileName = sprintf('slice_%d_time_series.gif', sliceOfInterest);

% Create figure for visualization
h = figure('Position', [1000, 600, 1200, 600]);
colormap gray;

% Loop through each time point (132 in total)
for timeIdx = 1:size(data, 4)
    % Extract selected slice for current time point
    dataSlice = data(:, :, sliceOfInterest, timeIdx);
    maskSlice = BWprop_LRT(:, :, sliceOfInterest, timeIdx);
    
    % Resize images and masks to be square for visualization
    targetSize = max(size(data, 1), size(data, 2));
    dataSliceSquare = imresize(dataSlice, [targetSize, targetSize]);
    maskSliceSquare = imresize(maskSlice, [targetSize, targetSize]);
    
    % Display image slice at current time point
    subplot(1, 2, 1);
    imagesc(dataSliceSquare);
    title(sprintf('Image - Slice %d at Time Point %d', sliceOfInterest, timeIdx));
    axis off;
    
    % Display corresponding mask slice
    subplot(1, 2, 2);
    imagesc(maskSliceSquare);
    title(sprintf('Mask - Slice %d at Time Point %d', sliceOfInterest, timeIdx));
    axis off;
    
    % Capture current frame for GIF
    frame = getframe(h);
    img = frame2im(frame);
    [imind, cm] = rgb2ind(img, 256);
    
    % Write to GIF file
    if timeIdx == 1
        imwrite(imind, cm, gifFileName, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
    else
        imwrite(imind, cm, gifFileName, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end
    
    pause(0.05); % Adjust pause time for visualization speed
end

disp(['GIF saved as ', gifFileName]);
