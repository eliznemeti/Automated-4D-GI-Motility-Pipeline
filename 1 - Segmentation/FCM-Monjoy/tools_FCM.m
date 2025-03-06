function [L,C,U,LUT,H]=tools_FCM(im,c,q)
% Segment N-dimensional grayscale image into c classes using a memory 
% efficient implementation of the fuzzy c-means (FCM) clustering algorithm. 
% The computational efficiency is achieved by using the histogram of the 
% image intensities during the clustering process instead of the raw image
% data.
%
% INPUT ARGUMENTS (3):
%   - im  : N-dimensional grayscale image in integer format. 
%   - c   : positive interger greater than 1 specifying the number of
%           clusters. c=2 is the default setting. Alternatively, c can be
%           specified as a k-by-1 array of initial cluster (aka prototype)
%           centroids.
%   - q   : fuzzy weighting exponent. q must be a real number greater than
%           1.1. q=2 is the default setting. Increasing q leads to an
%           increased amount of fuzzification, while reducing q leads to
%           crispier class memberships.
%
% OUTPUT (5):
%   - L   : label image of the same size as the input image. For example,
%           L==i represents the region associated with prototype C(i),
%           where i=[1,k] (k = number of clusters).
%   - C   : 1-by-k array of cluster centroids.
%   - U   : L-by-k array of fuzzy class memberships, where k is the number
%           of classes and L is the intensity range of the input image, 
%           such that L=numel(min(im(:)):max(im(:))).
%   - LUT : L-by-1 array that specifies the defuzzified intensity-class 
%           relations, where L is the dynamic intensity range of the input 
%           image. Specifically, LUT(1) corresponds to class assigned to 
%           min(im(:)) and LUT(L) corresponds to the class assigned to
%           max(im(:)). See 'apply_LUT' function for more info.
%   - H   : image histogram. If I=min(im(:)):max(im(:)) are the intensities
%           present in the input image, then H(i) is the number of image 
%           pixels/voxels that have intensity I(i). 
%
% AUTHOR    : Anton Semechko (a.semechko@gmail.com)
% DATE      : Apr.2013
%
% AVAILABLE FROM:
% https://www.mathworks.com/matlabcentral/fileexchange/41967-fast-segmentation-of-n-dimensional-grayscale-images?focused=3785642&tab=function


% Default input arguments
if nargin<2 || isempty(c), c=2; end
if nargin<3 || isempty(q), q=2; end

% Basic error checking to ensure valid inputs are provided
if nargin<1 || isempty(im)
    error('Insufficient number of input arguments')
end
msg='Revise variable used to specify class centroids. See function documentaion for more info.';
if ~isnumeric(c) || ~isvector(c)
    error(msg)
end
if numel(c)==1 && (~isnumeric(c) || round(c)~=c || c<2)
    error(msg)
end
if ~isnumeric(q)|| q<1.1 || numel(q)~=1
    error('Third input argument must be a real number > 1.1')
end

% checks input image is in an integer format and does not contain NaNs/Inf values
if isempty(strfind(class(im),'int'))
    error('Input image must be specified in integer format (e.g. uint8, int16)')
end
if sum(isnan(im(:)))~=0 || sum(isinf(im(:)))~=0
    error('Input image contains NaNs or Inf values. Remove them and try again.')
end

% Intensity range
Imin=double(min(im(:)));
Imax=double(max(im(:)));
I=(Imin:Imax)';

% Compute intensity histogram (histogram counts the number of pixels for each intensity level)
H=hist(double(im(:)),I);
H=H(:);
[junk,C]=tools_KM(im,c); % calling K means function for initial clustering (aka first guess) (k means typical for this)

%% Main FCM algorithm for fuzzy membership optimization 

% Update fuzzy memberships and cluster centroids
    % U -- fuzzy memberships
    % C -- cluster centroids
    % dC -- change in centroids
I=repmat(I,[1 c]); dC=Inf;
while dC>1E-6 % chosen threshold

    C0=C;
    
    % Distance to the centroids (corresponds to Euclidean distance ∣∣xi​−yj​∣∣ in FCM objective func)
    D=abs(bsxfun(@minus,I,C));
    D=D.^(2/(q-1))+eps;
    
    % Compute fuzzy memberships (corresponds to uij values from the FCM objective func)
    U=bsxfun(@times,D,sum(1./D,2));
    U=1./(U+eps);
    
    % Update the centroids
    UH=bsxfun(@times,U.^q,H);
    C=sum(UH.*I,1)./sum(UH,1);
    C=sort(C,'ascend'); % enforce natural order
    
    % Change in centroids 
    dC=max(abs(C-C0));
    
end

% Defuzzify and create a label image (converting the fuzzy classification into a crisp classification after convergence)
[Umax,LUT]=max(U,[],2);
L=LUT2label(im,LUT);

function L=LUT2label(im,LUT)
% Create a label image using LUT obtained from a clustering operation of 
% grayscale data.  

% Intensity range
Imin=min(im(:));
Imax=max(im(:));
I=Imin:Imax;

% Create label image
L=zeros(size(im),'uint8');
for k=1:max(LUT)
    
    % Intensity range for k-th class
    i=find(LUT==k);
    i1=i(1);
    if numel(i)>1
        i2=i(end);
    else
        i2=i1;
    end
    
    % Map the intensities in the range [I(i1),I(i2)] to class k 
    bw=im>=I(i1) & im<=I(i2);
    L(bw)=k;
    
end
