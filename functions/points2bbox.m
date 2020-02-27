% A function that takes as input 4 points of a rectangle in a 4-by-2 matrix 
% and returns them in bbox format. The inputs points are expected to be 
% ordered points of a rectangle, starting from the top left point then going clockwise
% PAY ATTENTION: the function expects a rectangle with the sides
% parallel to the x and y axis. If not, the function will return a wrong
% result. 
function bbox = points2bbox(points)

% check well formatted input as 4 points of a rectangle
% cond = points(2,1)>points(1,1) && points(1,2)==points(2,2) && points(3,2)>points(2,2) && ...
%    points(3,1)==points(2,1) && points(4,2)==points(3,2) && points(4,1)==points(1,1);
% assert(cond , 'Error in input argument (wrong format)');
bbox(1,1) = points(1,1);
bbox(1,2) = points(1,2);
bbox(1,3) = norm(points(1,:)-points(2,:));
bbox(1,4) = norm(points(1,:)-points(4,:));

end