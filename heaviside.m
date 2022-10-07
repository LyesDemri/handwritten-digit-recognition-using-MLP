function Y = heaviside(X)

%the heaviside function is the derivative for the ReLU activation function
Y = floor((sign(X)+1)/2);