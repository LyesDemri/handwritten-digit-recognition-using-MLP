function [O,Y] = forward(X,W2,b2,W1,b1,layers)

%3 activation functions are supported for both layers
%For some reason only sigmoid activation functions seem to give any
%interesting results
if strcmp(layers{1},'ReLU')
    Y = ReLU(W1*X+b1);
elseif strcmp(layers{1},'purelin');
    Y = W1*X+b1; 
elseif strcmp(layers{1},'sigmoid');
    Y = sigmoid(W1*X+b1);
else
    error(['unknown activation function ' layers{1}]);
end

if strcmp(layers{2},'ReLU')
    O = ReLU(W2*Y+b2);
elseif strcmp(layers{2},'purelin');
    O = W2*Y+b2;
elseif strcmp(layers{2},'sigmoid');
    O = sigmoid(W2*Y+b2);
else
    error(['unknown activation function ' layers{2}]);
end