clear;clc;close all;

%This script can be used to recognize numbers.
%You can use Paint or any application to draw numbers
%You have to draw the number in white on a black background
%Set the image so that it's 20x20 pixels
%Save the image in the correct directory then run this script
%The answer printed in the command window should be whatever number you
%typed in.
%I'm providing the best model I've trained so you can test this
load('model_784_100_10_97percent.mat');

I = imread('number.png');
I = double(rgb2gray(I));
A = zeros(28,28);
A(5:24,5:24) = I;
I=A;

X = reshape(I,28*28,1);
X = X/255;
O = forward(X,finalW2,finalb2,finalW1,finalb1,layers);

[osf,answer] = max(O);
answer = answer-1