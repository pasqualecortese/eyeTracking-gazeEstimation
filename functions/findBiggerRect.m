function rect = findBiggerRect (rect_array)
%find bigger rectangle in a rectangle array
    %initialization
    rect = rect_array(1,:);
    max_area = rect(3)*rect(4);
    %loop
    if size(rect_array,1)>1
        for i = 2:size(rect_array,1)
            if rect_array(i,3)*rect_array(i,4) > max_area
                max_area = rect_array(i,3)*rect_array(i,4);
                rect = rect_array(i,:);
            end
        end
    end
end