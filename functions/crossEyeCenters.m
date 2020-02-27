function [ioc_dist,theta] = crossEyeCenters(eyeCenter_left,eyeCenter_right)
    leftPoint = [eyeCenter_left,1]';
    rightPoint = [eyeCenter_right,1]';
    ioc_dist = norm(leftPoint-rightPoint);
    x = eyeCenter_right-eyeCenter_left;
    theta = atan2(x(2),x(1));
end