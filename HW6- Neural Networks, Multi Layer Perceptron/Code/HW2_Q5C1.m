clc;
clear all;

load('HW2_Data.mat')

mu = 0; sigma = 1; size_letters = 784; 
k_count = 10; %class count / output units count

train_images_squared = train_imgs.*train_imgs;
train_bias = ones(60000,1);
train_imgs = [train_imgs train_bias]; 

test_bias = ones(10000,1);
test_imgs = [test_imgs test_bias]; 

train_imgs_T = train_imgs';
test_imgs_T = test_imgs';

nu = 10^(-2);

train_labels_one_hot = zeros(10,60000);
for i = 1:60000
    train_labels_one_hot(train_labels(i) + 1,i) = 1;
end

% Homogenous coordinates, b should be added
bias_input = ones(10, 1);
output_weights = normrnd(mu, sigma, [10 size_letters]);
W1 = [output_weights bias_input];

Epochs = 15;
errors_train = zeros(1,Epochs+1);
errors_test = zeros(1,Epochs+1);

err1 = calc_err(W1, train_imgs, train_labels);
err2 = calc_err(W1, test_imgs, test_labels);
errors_train(1,1) = err1;
errors_test(1,1) = err2;

for Epoch = 1:Epochs
    % Backpropogation loop
    for loop = 1:60000
        x = train_imgs(loop,:);
        % **** FORWARD STEP ****
        y = W1 * x';
        z = exp(y) / sum(exp(y),1); %Softmax

        sens1 = train_labels_one_hot(:,loop) - z; 
        Grad1 = sens1*x;
        W1 = W1 + nu * Grad1;
    end

      err1 = calc_err(W1,train_imgs, train_labels);
      err2 = calc_err(W1,test_imgs, test_labels);
      errors_train(1,Epoch+1) = err1;
      errors_test(1,Epoch+1) = err2;
end

figure()
plot(0:Epochs, errors_train, 'b')
hold on
plot(0:Epochs, errors_test, 'r')
hold off
xlabel('Epochs');
xticks(0:Epochs)
ylabel('Prob of error')
legend('Train','Test')
title('SGD, Single-Layer (Softmax)')
sprintf('SGD, Single Layer(Softmax), Train err: %f, Test err: %f', errors_train(1,16),errors_test(1,16))

function err = calc_err(W1,x,labels)
    y = W1 * x';
    z = exp(y) ./ sum(exp(y),1); %Softmax
    
    [values, indices] = max(z);
    indices = indices - 1;
    err = sum(sum((labels ~= indices'))) / size(x,1);
end