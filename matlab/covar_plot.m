clear all;
clc;

load hls;

index = [15,16,21,37,43,48,57,60,62,64];

for i=1:10
    for j=(i+1):10
        figure();
        plot(s(:,index(i)), s(:,index(j)),'o')
    end
end