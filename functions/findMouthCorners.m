function mouthCorners = findMouthCorners(mouthImage)
    mouthCorners = zeros(2); %#ok<PREALL>
    mouthImage = im2double(mouthImage);
    R = mouthImage(:,:,1);
    G = mouthImage(:,:,2);
    chrom = zeros(size(mouthImage,1),size(mouthImage,2));
    for ii=1:size(mouthImage,1)
        for jj=1:size(mouthImage,2)
            aus = (R(ii,jj)-G(ii,jj))/R(ii,jj);
            chrom(ii,jj) = 2*atan(aus)/pi;
        end
    end
    threshold1 = mean(mean(chrom));
    for ii=1:size(mouthImage,1)
        for jj=1:size(mouthImage,2)
            if chrom(ii,jj)>threshold1
                chrom(ii,jj) = 255;
            else
                chrom(ii,jj) = 0;
            end
        end
    end

    threshold2 = 4000;
    IPx = sum(chrom,1);
    x1 = find(IPx>threshold2,1);
    x2 = length(IPx)-find(flip(IPx)>threshold2,1);
    y1 = mean(find(chrom(:,x1)>0));
    y2 = mean(find(chrom(:,x2)>0));
    
    mouthCorners = [x1,y1;x2,y2];
    %figure(2),imshow(chrom)
end

