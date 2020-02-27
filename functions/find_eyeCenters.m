function [eyeCenters,bbox_eyes] = find_eyeCenters(bboxFace,videoFrame,videoFrameGray)
    eyeCenters = zeros(2);
    bbox_eyes = zeros(2,4);
    %coordinates of the center of the box
    x2_dim = ceil(bboxFace(1)+bboxFace(3)/2);
    y2_dim = ceil(bboxFace(2)+bboxFace(4)/2);
    if (y2_dim<=size(videoFrame,1)) && (x2_dim<=size(videoFrame,2)) &&...
        (bboxFace(2)+bboxFace(4)<=size(videoFrame,1)) && (bboxFace(1)+bboxFace(3)<=size(videoFrame,2))
        for index=1:2 %(1=left,2=right)
            if index==1
                faceFrameGray = videoFrameGray(bboxFace(2):y2_dim,bboxFace(1):x2_dim);
                bbox_eye_left = leftEyeDetector.step(faceFrameGray);
                if ~isempty(bbox_eye_left)
                    bbox_eye_left = bbox_eye_left(1,:);
                    bbox_eye_left(1:2) = bbox_eye_left(1:2)+bboxFace(1:2);
                    eyeImage = videoFrame(bbox_eye_left(2):bbox_eye_left(2)+bbox_eye_left(4),...
                        bbox_eye_left(1):bbox_eye_left(1)+bbox_eye_left(3),:);
                    coord = eyecenter_loc(eyeImage,1);
                    eyeCenters(1,:) = coord+bbox_eye_left(1:2);
                    bbox_eyes(1,:) = bbox_eye_left;
                end
            else
                faceFrameGray = videoFrameGray(y2_dim:bboxFace(2)+bboxFace(4),...
                    x2_dim:bboxFace(1)+bboxFace(3));
                bbox_eye_right = rightEyeDetector.step(faceFrameGray);
                if ~isempty(bbox_eye_right)
                    bbox_eye_right = bbox_eye_right(1,:);
                    bbox_eye_right(1:2) = bbox_eye_right(1:2)+bboxFace(1:2);
                    eyeImage = videoFrame(bbox_eye_right(2):bbox_eye_right(2)+bbox_eye_right(4),...
                        bbox_eye_right(1):bbox_eye_right(1)+bbox_eye_right(3),:);
                    coord = eyecenter_loc(eyeImage,1);
                    eyeCenters(2,:) = coord+bbox_eye_right(1:2);
                    bbox_eyes(2,:) = bbox_eye_right;    
                end
            end
        end
    end   
end