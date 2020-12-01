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

nu = 10^(-4);

train_labels_one_hot = zeros(10,60000);
for i = 1:60000
    train_labels_one_hot(train_labels(i) + 1,i) = 1;
end


H_Values = [10, 20, 50];
for H_Layer = H_Values

    % Homogenous coordinates, b should be added
    bias_input = ones(H_Layer, 1);

    hidden_weights = normrnd(mu, sigma, [H_Layer size_letters]);
    W1 = [hidden_weights bias_input]; % Homogenous weights, input bias term added
    W2 = normrnd(mu, sigma, [k_count H_Layer]);

    % Backpropogation loops initialization
    checkpoints = 200;
    loop_count = 2000;
    loop_w_hidden = zeros(size(W1,1), size(W1,2), checkpoints);
    loop_w_output = zeros(size(W2,1), size(W2,2), checkpoints);
    loop_w_hidden(:,:,1) = W1;
    loop_w_output(:,:,1) = W2;

    count = 0;
    % Backpropogation loop
    for loop = 1:loop_count
        % **** FORWARD STEP ****
        g = W1 * train_imgs';
        y = 1 ./ (1 + exp(-g));
        u = W2 * y;
        z = exp(u) ./ sum(exp(u),1); %Softmax

        Grad2 = (train_labels_one_hot - z)*y'; 

    %   % The longer, loop solution
    %     Grad1 = zeros(15,785);  
    %     for n = 1:60000
    %         %Grad1_1 = (train_labels_one_hot(:,n) - z(:,n))'*W2;
    %         %Grad1_2 = (y(:,n) .* (1 - y(:,n)))  * train_imgs(n,:);
    %         %Grad1 = Grad1 + diag((train_labels_one_hot(:,n) - z(:,n))'*W2) * (y(:,n) .* (1 - y(:,n)))  * train_imgs(n,:);
    %     end

        % Faster solution, same operation as above
        Grad1 = ((((train_labels_one_hot - z)'*W2)').*(y .* (1 - y)))*train_imgs;
        W1 = W1 + nu * Grad1;
        W2 = W2 + nu * Grad2;

        if rem(loop,loop_count/checkpoints) == 0
            count = count + 1;
            loop_w_hidden(:,:,count) = W1;
            loop_w_output(:,:,count) = W2;
        end
    end

    errors_train = zeros(1,checkpoints);
    errors_test = zeros(1,checkpoints);
    for loop = 1:checkpoints
        % **** FORWARD STEP ****
        W1 = loop_w_hidden(:,:,loop);
        W2 = loop_w_output(:,:,loop);
        err1 = calc_err(W1,W2,train_imgs,train_labels);
        err2 = calc_err(W1,W2,test_imgs,test_labels);
        errors_train(1,loop) = err1;
        errors_test(1,loop) = err2;
    end

    figure()
    plot(linspace(1,loop_count,checkpoints), errors_train, 'b')
    hold on
    plot(linspace(1,loop_count,checkpoints), errors_test, 'r')
    hold off
    xlabel('Iterations');
    ylim([0 inf]);
    %xticks(0:Epochs)
    ylabel('Prob of error')
    legend('Train','Test')
    title(sprintf('Q5B)ii Batch Learning, Sigmoid H = %d', H_Layer))
    sprintf('Q5B)ii Batch Learning, Sigmoid, H = %d: Train err: %f, Test err: %f', H_Layer, errors_train(1,checkpoints),errors_test(1,checkpoints))
    
end

function err = calc_err(W1,W2,data,labels)
    g = W1 * data';
    y = 1 ./ (1 + exp(-g));
    u = W2 * y;
    z = exp(u) ./ sum(exp(u),1); %Softmax
    
    [values, indices] = max(z);
    indices = indices - 1;
    err = sum(sum((labels ~= indices'))) / size(data,1);
end