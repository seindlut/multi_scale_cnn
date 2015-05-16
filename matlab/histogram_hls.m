clear all;
clc;

load hls;

for i=1:size(s, 2)
    figure();
	hist(s(:,i), 20);
end
