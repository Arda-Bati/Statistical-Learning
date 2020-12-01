clc;
clear all;
trainImg = 'train-images.idx3-ubyte';
trainLabel = 'train-labels.idx1-ubyte';

testImg = 't10k-images.idx3-ubyte';
testLabel = 't10k-labels.idx1-ubyte';

[train_imgs, train_labels] = readMNIST(trainImg, trainLabel, 60000, 0);
[test_imgs, test_labels] = readMNIST(testImg, testLabel, 10000, 0);

save HW2_Data
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

nu = 10^(-5);

train_labels_one_hot = zeros(10,60000);
for i = 1:60000
    train_labels_one_hot(train_labels(i) + 1,i) = 1;
end

% Homogenous coordinates, b should be added
weights = normrnd(mu, sigma, [k_count size_letters]);
bias = ones(10, 1);
w_all = [weights bias]; % Homogenous weights, input bias term added
loop_count = 1000;
loop_w_all = zeros(10,785,loop_count);
loop_w_all(:,:,1) = w_all;

for loop = 1:loop_count
    %Forward step
    cur_w = loop_w_all(:,:,loop);
    actvs = cur_w * train_imgs_T;
    % Softmax calculation
    y_values = exp(actvs) ./ sum(exp(actvs),1);
    % Descent step calculation
    Grad_E = (train_labels_one_hot - y_values)*(train_imgs);
    % Updating weights
    loop_w_all(:,:,loop + 1) = cur_w + nu * Grad_E;
end

errors_train = zeros(1,loop_count);
errors_test = zeros(1,loop_count);

for loop = 1:loop_count    
    %Forward step
    cur_w = loop_w_all(:,:,loop);
    err1 = calc_err(cur_w,train_imgs,train_labels);
    err2 = calc_err(cur_w,test_imgs,test_labels);
    errors_train(loop) = err1;
    errors_test(loop) = err2;
end

figure()
loop = [1:loop_count];
plot(loop, errors_test, 'r');
hold on
plot(loop, errors_train, 'b');
ylim([0.05, inf]);
title('Bacth Learning, Single Layer (Softmax)');
xlabel('#Iterations');
ylabel('Prob of error')
legend('Train','Test')
sprintf('Batch Learning, Single Layer(Softmax), Train err: %f, Test err: %f', errors_train(1,loop_count),errors_test(1,loop_count))

function err = calc_err(W1,data,labels)
    actvs = W1 * data';
    % Softmax results
    y_values = exp(actvs) ./ sum(exp(actvs),1);
    [values, indices] = max(y_values);
    indices = indices - 1;
    % Error calculation
    err = sum(sum((labels ~= indices'))) / size(data,1);
end