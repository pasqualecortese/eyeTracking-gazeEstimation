function eyeCenter = eyecenter_loc(eyeImage)
% EYE CENTER LOCALIZATION: 

    % RGB -> YCbCr color space
    eyeImageYcbcr = rgb2ycbcr(eyeImage);
    % Y, Cb, Cr channels are isolated 
    Y =  mat2gray(eyeImageYcbcr(:,:,1)); 
    Cb = mat2gray(eyeImageYcbcr(:,:,2));
    Cr = mat2gray(eyeImageYcbcr(:,:,3));

    % EyeMapC
    cbCrDivision = Cb./Cr;
    mapInf = isinf(cbCrDivision);
    % we want to saturate the cbCrDivision to the maximum number (infinity excluded)
    cbCrDivision(mapInf) = 0; % this is done to return the maximum number, otherwise we would get 'inf' as the max
    cbCrDivision(mapInf) = max(cbCrDivision, [], 'all');
    % in ther case of 0/0 divion we get NaN value and we substitute
    % it with 0
    cbCrDivision(isnan(cbCrDivision)) = 0; 
    eyeMapC = 1/3 * ( Cb.^2 + (1-Cr).^2 + cbCrDivision );

    % EyeMapI
    eyeRegionWidth = size(eyeImage,2);
    eyeRegionHeight = size(eyeImage,1);
    irisRad = eyeRegionWidth/10;            
    % Dilation of eyeMapC with the flat circular structuring
    % element B1
    B1Rad = floor(irisRad/2);
    B1 = strel('disk', double(B1Rad));
    eyeMapCDilated = imdilate(eyeMapC, B1);

    % Erosion of the luminance channel with the flat circular
    % structuring element B2
    B2Rad = floor(irisRad/2); % on the paper B2Rad = floor(B1Rad/2)
    B2 = strel('disk', double(B2Rad));
    YEroded = imerode(Y, B2);

    delta = mean(YEroded, 'all') ;
    eyeMapI = eyeMapCDilated ./ (YEroded + delta);

    % Fast radial symmetry computation
    n_min = ceil(irisRad/2);
    n_max = ceil(2*irisRad); % the paper proposes 5*irisRad
    radii = n_min : n_max;
    alpha = 2; % a higher alpha eliminates nonradially symmetric features such as lines;  choosing alpha=1 minimizes the computation
    beta = 0.1; % we ignore small gradients by introducing a gradient threshold parameter beta

    S_luminance = mat2gray(-fastradial(YEroded, radii, alpha, beta, 'dark', 0));
    S_eyeMapI = mat2gray(fastradial(eyeMapI, radii, alpha, beta, 'bright', 0));

    sumS = S_luminance + S_eyeMapI;
    [maxvalue, argmax] = max(sumS(:));
    [max_row, max_col] = ind2sub(size(sumS), argmax);

    eyeCenter = [max_col, max_row];
end