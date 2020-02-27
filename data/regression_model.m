%% initialization
close all
clear
clc

%% preparation
files = dir('dataset*');
load(files(1).name)
if size(files,1)>1
    for ii=2:size(files,1)
        L = load(files(ii).name);
        calibrationTargets = [calibrationTargets;L.calibrationTargets];
        horizontal = [horizontal;L.horizontal];
        vertical = [vertical;L.vertical];
    end
end

%% regression
Xh = [ones(size(horizontal, 1), 1) horizontal(:,1) horizontal(:,2) horizontal(:,3)...
    horizontal(:,4) horizontal(:,5) horizontal(:,6)];
Xv = [ones(size(vertical, 1), 1) vertical(:,1) vertical(:,2) vertical(:,3)...
    vertical(:,4) vertical(:,5) vertical(:,6)];
Yh = calibrationTargets(:,1);
Yv = calibrationTargets(:,2);
             
% horizontal model parameters
H = regress(Yh,Xh);
% vertical model parameters
V = regress(Yv,Xv);

%% results
predH = zeros(size(calibrationTargets,1),1);
predV = zeros(size(calibrationTargets,1),1);
for ii = 1:size(calibrationTargets,1)
    predH(ii) = [1, horizontal(ii,:)]*H;
    predV(ii) = [1, vertical(ii,:)]*V;
end
subplot(211)
plot(Yh,'k'), hold on, plot(predH,'r')
title('horizontal ax'), legend('real','predicted')
subplot(212)
plot(Yv,'k'), hold on, plot(predV,'r')
title('vertical ax'), legend('real','predicted')

%% closing and saving
clearvars -except H V
save('regressionModel.mat')