clear all;
clc;

load hls;
for i=1:64
    figure()
    hist(s(:,i));
end