% Path to your file
niftiFile = '/Users/elizabethnemeti/Desktop/Subject_reconstruction_comparisons/FD_027_4D_reconstructed_original.nii';
niftiData = load_nii(niftiFile);

% Extract the 4D MRI volume
stomachMask = niftiData.img; % Binary mask (1 = stomach, 0 = background)
[numRows, numCols, numSlices, numTimeFrames] = size(stomachMask);
disp('Size of stomachMask:');
disp(size(stomachMask)); % Should output [256, 256, Slices, TimeFrames]

%%%%%% Cross-Sectional Area Changes %%%%%%
crossSectionalAreas = zeros(numSlices, numTimeFrames);

for sliceIdx = 1:numSlices
    for t = 1:numTimeFrames
        % Sum all non-zero pixels in each slice
        currentSlice = squeeze(stomachMask(:, :, sliceIdx, t));
        crossSectionalAreas(sliceIdx, t) = sum(currentSlice(:));
    end
end

%%%%%% Frequency Analysis %%%%%%
Fs = 1; % Sampling rate (frames per second)
physiologicalFreqRange = [0.05, 0.2]; % Hz range for gastric peristalsis

% Analyze mid-slice cross-sectional area changes
midSliceAreas = crossSectionalAreas(ceil(numSlices / 2), :);
detrendedData = midSliceAreas - mean(midSliceAreas); % Detrend data

% Apply FFT
Y = fft(detrendedData);
f = (0:(numTimeFrames - 1)) * Fs / numTimeFrames;
amplitudeSpectrum = abs(Y);

% Filter to physiological range
frequencyMask = (f >= physiologicalFreqRange(1) & f <= physiologicalFreqRange(2));
filteredAmplitudes = amplitudeSpectrum .* frequencyMask;

% Identify dominant frequency
[~, dominantFreqIdx] = max(filteredAmplitudes);
dominantFreq = f(dominantFreqIdx);
disp('Dominant Frequency (Hz):');
disp(dominantFreq);

%%%%%% Wave Propagation Speed %%%%%%

% Cross-correlation between slices
waveSpeeds = zeros(1, numSlices - 1);
for sliceIdx = 1:(numSlices - 1)
    slice1 = detrend(crossSectionalAreas(sliceIdx, :));
    slice2 = detrend(crossSectionalAreas(sliceIdx + 1, :));
    
    [xc, lags] = xcorr(slice1, slice2, 'coeff');
    [~, maxIdx] = max(xc); % Find the lag with max correlation
    timeLag = lags(maxIdx) / Fs; % Time delay in seconds
    
    % Convert lag to speed (distance per time)
    distanceBetweenSlices = 5; % mm 
    waveSpeeds(sliceIdx) = distanceBetweenSlices / timeLag; % mm/s
end

disp('Wave propagation speeds (mm/s):');
disp(waveSpeeds);

%%%%%% Visualization %%%%%%
% 3D Visualization of Cross-Sectional Area with Time on X-Axis
figure;
[X, Y] = meshgrid(1:numSlices, 1:numTimeFrames); % Swap X and Y
surf(Y, X, crossSectionalAreas', 'EdgeColor', 'none'); % Transpose data to align axes
xlabel('Time Frame');
ylabel('Slice Index');
zlabel('Cross-Sectional Area (Pixels)');
title('3D Visualization of Peristalsis');
colormap jet;
colorbar;

%%%%%% Slice 35 Visualization %%%%%%
sliceToVisualize = 35; % Slice index to visualize

% Extract the cross-sectional area for the selected slice
slice35Areas = crossSectionalAreas(sliceToVisualize, :);

% Plot the changes in cross-sectional area over time
figure;
plot(1:numTimeFrames, slice35Areas, '-o', 'LineWidth', 2, 'MarkerSize', 5);
xlabel('Time Frame');
ylabel('Cross-Sectional Area (Pixels)');
title(sprintf('Cross-Sectional Area Changes for Slice %d', sliceToVisualize));
grid on;
