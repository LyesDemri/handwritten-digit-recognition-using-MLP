clear;clc;close all;

%You get to continue training from last training in case you want to
%Enter C to continue from last save point, or A if you want to start anew
answer = input('Continue (C) or start anew (A)? ','s');
if strcmpi(answer,'c')
    %loads all variables from last session
    load('state.mat');
    continuing = true;  %so we know not to generate a new random network
    i=1;    %or else it will be 60000
    disp(['Continuing from epoch ' num2str(epoch) ' example ' num2str(i)]);
    %plot learning error history for all epochs so far
    figure(1);plot(errors_list,'-o'); xlim([0,N_epochs]);ylim([0,60000]);grid on;
    pause(1)    %gives the time for the figure to display
elseif strcmpi(answer,'a')
    disp('Starting over from scratch');
    continuing = false;
else
    error('Invalid answer!')
end

%start timing
tic
if ~continuing
    total_duration = 0;
    %model:
    input_layer_units = 28*28;
    hidden_layer_units = 100;    %you can play around setting this
    %too many hidden layer units will overlearn
    %(100 might be too much already)
    output_layer_units = 10;
    W1 = rand(hidden_layer_units,input_layer_units)*2 - 1;
    b1 = rand(hidden_layer_units,1)*2 - 1;
    W2 = rand(output_layer_units,hidden_layer_units)*2 - 1;
    b2 = rand(output_layer_units,1)*2 - 1;
    %There are 3 possible activation functions: purelin, sigmoid, and ReLU 
    %only the sigmoid/sigmoid combination gives any good result, not sure 
    %if that's how it's normal
    layers = {'sigmoid','sigmoid'};

    %data
    load train-images-idx3-ubyte.mat
    load train-labels.idx1-ubyte.mat
    %input images are flattened as 60000 columns of 784 values
    X = reshape(pixel,28*28,1,size(pixel,3)); clear pixel;
    X = reshape(X,28*28,size(X,3));
    %normalize input to [0,1]
    X = X/255;
    %target vectors are one-hot encoded
    T = zeros(10,length(label));
    for i = 1:size(label,2)
        T(label(i)+1,i) = 1;
    end
    %discard voluminous unneeded variables
    clear label;

    i=1;
    epoch = 1;
    errors = 0;
    %N_epochs should be sufficient (probably above 90% good recognition)
    N_epochs = 100;
    %this is to plot learning error history
    %(number of errors for each epoch)
    errors_list = 0;
    %randomize order of presentation of examples
    example_list = randperm(length(X));
end
%I like to hear a small beep sound when each epoch is complete:
y = sin(2*pi*880*(0:1/16000:0.1));
while epoch <= N_epochs
    %forward:
    [O,Y] = forward(X(:,example_list(i)),W2,b2,W1,b1,layers);

    %get network answer for this input
    [whatever,answer] = max(O); 
    answer = answer - 1;
    %get correct answer from T matrix:
    [whatever,correct_answer] = max(T(:,example_list(i)));
    correct_answer = correct_answer - 1;
    %compare results
    if answer ~= correct_answer
        errors = errors+1;
    end

    E = (T(:,example_list(i))-O).^2;
    %backward:
    %You have to know how to compute these by hand
    %Pretty simple most of the time when you know the chain rule:
    %dE/dW2 = dE/d(T-O)         x ... 2(T-O)
    %         %d(T-O)/d(O)      x ... -1
    %         %d(O)/d(W2*Y+b2)  x ... O*(1-O)   (for sigmoid activation)
    %         %d(W2*Y+b2)/d(W2) x ... Y
    %dE/db2 = dE/d(T-O)         x ... 2(T-O)
    %         %d(T-O)/d(O)      x ... -1
    %         %d(O)/d(W2*Y+b2)  x ... O*(1-O)   (for sigmoid activation)
    %         %d(W2*Y+b2)/d(b2) x ... 1
    %The derivative is different for different activation functions
    if strcmp(layers{2},'ReLU')
        delta2 = 2*(O-T(:,example_list(i))).*heaviside(O);
    elseif strcmp(layers{2},'sigmoid')
        delta2 = 2*(O-T(:,example_list(i))).*O.*(1-O);
    elseif strcmp(layers{2},'purelin')
        delta2 = 2*(O-T(:,example_list(i)));
    else
        error(['unknown layer type: ' layers{2}])
    end
    
    %do the same thing for first layer
    %I know this could be done with a loop, but it's not the point of this
    %program
    %dE/dW1 = dE/d(T-O)         x ... 2(T-O)
    %         %d(T-O)/d(O)      x ... -1
    %         %d(O)/d(W2*Y+b2)  x ... O*(1-O)   %for sigoid activation
    %         %d(W2*Y+b2)/d(Y)  x ... W2
    %         %d(Y)/d(W1*X+b1)  x ... heaviside(Y)
    %         %d(W1*X+b1)/W1    x ... X
    if strcmp(layers{1},'ReLU')
        delta1 = W2'*delta2.*heaviside(Y); %using ReLU
    elseif strcmp(layers{1},'sigmoid')
        delta1 = W2'*delta2.*(Y.*(1-Y));    %using sigmoid
    elseif strcmp(layers{1},'purelin')
        delta1 = W2'*delta2;    %using purelin
    else
        error(['unknown layer type: ' layers{2}])
    end
    
    %complete computation of gradients
    deltaW2 = delta2*Y';
    deltab2 = delta2;
    deltaW1 = delta1*(X(:,example_list(i)))';
    deltab1 = delta1;
    %update weights (learning rate arbitrarily set to 0.1, best value I've found)
    W2 = W2 - 0.1*deltaW2;
    b2 = b2 - 0.1*deltab2;
    W1 = W1 - 0.1*deltaW1;
    b1 = b1 - 0.1*deltab1;

    i = i+1;
    if i==length(T)
        %end of epoch
        total_duration = total_duration + toc;  %update duration
        tic
        save('state.mat');  %save data
        %it's not a good idea to save too often, slows down the
        %calculations a lot. I've found that saving every epoch was a good
        %compromise
        disp(['epoch: ' num2str(epoch) ', ' num2str(errors) ' errors, total duration: ' num2str(total_duration) ', state saved']);
        i=1;
        %add number of errors to learning errors history
        errors_list(epoch) = errors;
        if errors == min(errors_list)
            %remember best parameters so far
            finalW2 = W2;
            finalb2 = b2;
            finalW1 = W1;
            finalb1 = b1;
        end
        %This plots the learning error history
        figure(1);plot(errors_list,'-o');xlim([0,N_epochs]);ylim([0,60000]);grid on;
        title(['epoch ' num2str(epoch)])
        %if you don't pause the system to let the figure appear, it will
        %appear much later than it should:
        pause(1)
        epoch = epoch + 1;
        %play beep sound
        wavplay(y,16000,'async');
        %if recognition rate for this epoch is higher than 99%, stop
        %training. I set this to arbitrarily high values.
        %If the network's gonna learn in just 10 epochs, what's the point
        %of doing 100 epochs?
        if errors/size(X,2) < 0.01
            break;
            %sound to let you know the network has learned
            load gong;
            wavplay(y,Fs);
        end
        %reset errors counter and generate new example list
        errors = 0;
        example_list = randperm(length(X));
    end
end

disp(['errors:' num2str(min(errors_list))])

if epoch < N_epochs
    %if the system has learned before we've done the chosen number of
    %epochs, let the user know
    disp(['The model has learned after only ' num2str(epoch) ' epochs'])
end
disp('Done training! Now testing network performance')
%test:
%This lets the user see each example image and the network's answer to this
%example
%If you just want to know the validation rate, say N
%If you want to see the network's reponse to each input, say Y
user_input = input('Show performance live? (Y/N)','s');
if strcmpi(user_input,'Y')
    showing_performance = 1;
else
    showing_performance = 0;
end
disp('Testing...')
%load test data (same as for training)
load t10k_images_idx3_ubyte.mat
load t10k-labels.idx1-ubyte.mat
images = pixel; %I prefer this to be named "images" and not "pixel"
X = reshape(pixel,28*28,1,size(pixel,3)); clear pixel;
X = reshape(X,28*28,size(X,3));
X = X/255;
T = zeros(10,length(label));
for i = 1:size(label,2)
    T(label(i)+1,i) = 1;
end
clear label;
errors = 0;
%things are pretty straightforward from here, we're just counting the
%errors
for i = 1:length(X)
    O = forward(X(:,i),finalW2,finalb2,finalW1,finalb1,layers);
    [osf,answer] = max(O); answer = answer -1;
    [osf,correct_answer] = max(T(:,i));
    correct_answer = correct_answer - 1;
    if answer ~= correct_answer
        errors = errors + 1;
    end
    if showing_performance == 1
        figure(2)
        imshow(images(:,:,i)); 
        title(['Answer given by network: ' num2str(answer)])
        A = input('keep showing performances? (Y/N)','s');
        if strcmpi(A,'N')
            showing_performance = 0;
        end
    end
end

%display the results to the user:
disp(['Errors on test set: ' num2str(errors) '/' num2str(length(X))]);
disp(['Error rate on test set: ' num2str(errors*100/length(X)) '%']);

%You can use the next line to save the model:
%save('saved_model.mat','finalW2','finalW1','finalb2','finalb1','layers');
