clc;close all; clear all;

%% Values of the Variables
% parameters for the fuzzy c-means algorithm and active contouring
fuzziness = 3; % FCM -> moderate fuzz (helps segment where transition between different regions is not sharply defined)
numClust = 3; % FCM -> Cluster 1 = stomach, cluster 2 = other body organs, cluster 3 = background
winSize = 7; % AC -> defines window size around each pixel that AC considers when adjusting contours
lengthPenalty = 0.000001; % AC -> penalizes the length of the contour (highs penalty = smooth/simple vs low penalty = complex/jagged)
iteration = 400; % AC -> how many iterations alg will run for (more iterations = more refined/computation time) 
epsilon = 0.3; % AC -> convergence criterion (i.e. changed allowed between iterations, when changer falls below threshold, alg may assume it's congerged at a solution)
               %  small epsilon = more accurate/more iterations/slower vs large epsilon = less accurate/fewer iterations/quicker

% Update for the new dataset
base_path = '/Users/elizabethnemeti/Desktop/uw-madison-gi-tract-image-segmentation/train/case146/case146_day0/scans';
all_images = dir(fullfile(base_path, '*.png')); % Adjusted for PNG files

for ii = 1:length(all_images)
    %% Load PNG Image Data
    img_path = fullfile(base_path, all_images(ii).name);
    img_info = imfinfo(img_path); % Get image information to handle 16-bit images correctly
    img = imread(img_path);
    
    % Normalize and convert to intensity image
    img = mat2gray(img, [0 2^img_info.BitDepth-1]); % Adjust the range based on bit depth
    img = uint8(255 * img); % Convert from 16-bit to uint8 for processing
    
    %% 3 Stomach SEGMENTATION 
    [nx,ny,nz] = size(img); % returns dimensions of img as [nx, ny, nz] aka width, height, and depth, respectively
    output_temp = tools_FCM(img,numClust,fuzziness); % applies fuzzy c-means clustering to the image with specified numClust and fuzziness params
    img_fuzzy = (output_temp == numClust); % only keep the brightest class (binary pic)
    % Small objects are removed and holes are filled to clean up the segmentation
    img_fuzzy = bwareaopen(img_fuzzy,5,26); % remove objects smaller than 5 voxels
    img_fuzzy = imfill(img_fuzzy,26,'holes'); % fill holes
    %-- run 2D localized active contour (localized bc of small winsize)
    Output = zeros(nx,ny,nz); % initiliazes array of zeroes with dimensions of img, stores AC results for each slice of img
    ACPlot = figure('visible','on'); % figure window created for later AC visualization
    set(ACPlot, 'Position', [200,200, 1280,600],'color','w','name','Stomach Segmentation'); % settings for figure window
    
    %% Create folder to save results
    % Defining Output folder as 'Frames'
    opFolder = fullfile(cd, fname); % defines path to output folder (aka subdfolder in this directory)
    if ~exist(opFolder, 'dir') % checks if the folder specified in opFolder exists
    mkdir(opFolder); % create a new directory with path specified in opFolder (if not existing)
    end
    %% 

    for islice = 1 : 4 % loop over slices (islice indicates which slice we're on in the loop) 
        if sum(nnz(img_fuzzy(:,:,islice)))>0 % checking if current slice has relevant data (if sum of non zero elements > 0 = relevant data, if not = skip)
            figure(ACPlot);
            subplot(1,2,1);
            Output(:,:,islice) = ac(double(img(:,:,islice)),img_fuzzy(:,:,islice),winSize,lengthPenalty,iteration,epsilon); 
                % applies AC to current slice
                % double -> double precision convert)
                % img(:,:,islice) -> extracts one 2D slice from slice stack
                % img_fuzzy(:,:,islice) -> extracts one binary image (from FCM processing) from slice stack
            Output(:,:,islice) = imfill(Output(:,:,islice),'holes'); % imfill = fill holes in the binary image
            figure(ACPlot);
            subplot(1,2,2);
            imshowpair(imrotate(squeeze(img(:,:,islice)),90),imrotate(squeeze(Output(:,:,islice)),90)); % showing plots for 1. original slice next to 2. processed slice
            
            %% Save results in sub folders
            imwrite(imrotate(Output(:,:,islice), 90),fullfile(opFolder, [fname,'_bin_mask_',num2str(islice), '.png'])); % saves rotated binary mask as PNG
            % To save original image uncheck below line
            %imwrite(imrotate(img(:,:,islice),
            %90),fullfile(opFolder,[fname,'_',num2str(islice), '.png'])); % saves rotated original image slice as PNG
            maskedRgbImage = bsxfun(@times, (imrotate(img(:,:,islice), 90)), cast(imrotate(Output(:,:,islice),90), 'like', (imrotate(img(:,:,islice), 90)))); % creating the masked image
            % binary image is used to create masked image (by highlighting parts of the original image)
            % element-wise multiplication of mask with original image
            imwrite(maskedRgbImage,fullfile(opFolder, [fname,'_ori_mask_',num2str(islice), '.png'])); % saving the masked image 
            title(['Processing slice: ' num2str(islice)],'fontsize',18);
            pause(0.1);
        end        
    end
    close all
end