function [eyeLidsCoord] = findEyeLids(videoFrame,eyeCenter,ioc_dist,theta)
    eyeLidsCoord = zeros(2);
    ROI_width = ceil(ioc_dist*0.1/2);
    ROI_height = ceil(ioc_dist*0.3/2);
    ROI_area = videoFrame(eyeCenter(2)-ROI_height:eyeCenter(2)+ROI_height,...
        eyeCenter(1)-ROI_width:eyeCenter(1)+ROI_width,:);
    edges = edge(rgb2gray(ROI_area),'canny');
    IP = sum(edges,2);
    y1 = size(IP,1)/2-find(IP>0,1)*cos(theta);
    y2 = size(IP,1)/2-find(flip(IP)>0,1)*cos(theta);
    x1 = y1*sin(theta);
    x2 = y2*sin(theta);
    
    % no angle considered
    
    eyeLidsCoord = [eyeCenter(1)-x1,eyeCenter(2)-y1;eyeCenter(1)+x2,eyeCenter(2)+y2];
end