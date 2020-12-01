clc;
clear all;
trainImg = 'train-images.idx3-ubyte';
trainLabel = 'train-labels.idx1-ubyte';

testImg = 't10k-images.idx3-ubyte';
testLabel = 't10k-labels.idx1-ubyte';

[train_imgs, train_labels] = readMNIST(trainImg, trainLabel, 20000, 0);
[test_imgs, test_labels] = readMNIST(testImg, testLabel, 10000, 0);

save HW3_Data
%load('HW3_Data.mat')

digits_train = zeros(20000,10);
%Classifier for class digit = 0
for class_index = 0:9
    digits_train(:,class_index+1) = (train_labels == class_index)*2 - 1;
end

digits_test = zeros(10000,10);
%Classifier for class digit = 0
for class_index = 0:9
    digits_test(:,class_index+1) = (test_labels == class_index)*2 - 1;
end

train_errors = zeros(250,10);
test_errors = zeros(250,10);

wi_train = zeros(20000,10);

gt_train = zeros(20000,10);
gt_test = zeros(10000,10);
thresholds = repmat(linspace(0,1,51),1,2);

highest_weights = zeros(250,10);
margin = zeros(5,20000,10);

train_counts = zeros(10,1);
test_counts = zeros(10,1);

for i = 1:10
    train_counts(i) = sum(train_labels == i-1);
    test_counts(i) = sum(test_labels == i-1);
end

no_iter = 1;

tic
count = 0;
for iteration = 1:no_iter
    % Iteration will start here
    %iteration
    %Updating weights
    wi_train = exp(-(digits_train.*gt_train));
    
    [val, indx] = max(wi_train);
    highest_weights(iteration, :) = indx;
    
    if (ismember(iteration, [5, 10, 50, 100, 250]))
        count = count + 1;
        %margin = y*g(x)
        margin(count,:,:) = digits_train.*gt_train;
    end
 
    %Trying each possible weak learner
    for class_index = 1:10
        at = inf*ones(102,784);
        for i = 1:51
            threshold = thresholds(i);
            for dim = 1:784
                ux = (train_imgs(:,dim) >= threshold)*2 - 1;
                ux_polar = ux*(-1);
                %errs(i,dim) = (digits_train(:,class_index) ~= ux)'*wi_train(:,class_index);
                %errs(i+51,dim) = (digits_train(:,class_index) ~= ux_polar)'*wi_train(:,class_index);
                at(i,dim) = (digits_train(:,class_index).* ux)'*wi_train(:,class_index);
                at(i+51,dim) = (digits_train(:,class_index) .* ux_polar)'*wi_train(:,class_index);
            end
        end

        % Choosing the weak learner with the smallest error
        [~,idx] = max(at(:));
        [row,col]=ind2sub(size(at),idx);
        threshold = thresholds(row);

        % Step size calculation
        sgn = (row <= 51)*2 - 1; %Choosing the original or the polar
        ux_train = ((train_imgs(:,col) >= threshold)*2 - 1)*sgn;
        sum1_train = (digits_train(:,class_index) ~= ux_train)'*wi_train(:,class_index);
        sum2_train = sum(wi_train(:,class_index));
        E_train = sum1_train / sum2_train;
        wt = (1/2)*(log((1-E_train)/E_train));
        
        % Update
        gt_train(:,class_index) = gt_train(:,class_index) + wt*ux_train;
        
        % Train error calculations
        train_err  = (gt_train(:,class_index) >= 0) ~= (digits_train(:,class_index) >= 0);
        train_errors(iteration, class_index) = sum(train_err)/20000;
        
        % Test error calculations
        % Update

        ux_test = ((test_imgs(:,col) >= threshold)*2 - 1)*sgn;
        gt_test(:,class_index) = gt_test(:,class_index) + wt*ux_test;
        
        test_err = (gt_test(:,class_index) > 0) ~= (digits_test(:,class_index) > 0);
        test_errors(iteration, class_index) = sum(test_err)/10000;
    end
end
toc

train_errors(train_errors > 1) = 1;
test_errors(test_errors > 1) = 1;

[~, decisions] = max(gt_test,[],2);
decisions = decisions - 1;
err_rate = sum(test_labels ~= decisions) / size(decisions,1);
fprintf('The error rate of the final classifier on test set is %f: ',err_rate);

save HW3_Data_Loop_Complete

% Printing the results below

figure()
for i = 1:10
    subplot(2,5,i);
    plot(1:250,train_errors(:,i)/10,'b')
    hold on
    plot(1:250,test_errors(:,i)/10,'r')
    title(sprintf('Digit %d', i-1)); xlabel('Iteration t'); ylabel('Prob. of error.');
    legend('train','test');
    xticks(0:50:250)
end    

figure()
for i = 1:10
    subplot(2,5,i);
    hold on
    edges = linspace(-5,13,500);
    h1 = histogram(margin(1,:,i),500,'BinEdges',edges,'Normalization','cdf','DisplayStyle','stairs','EdgeColor','r');
    h2 = histogram(margin(2,:,i),500,'BinEdges',edges,'Normalization','cdf','DisplayStyle','stairs','EdgeColor','g');
    h3 = histogram(margin(3,:,i),500,'BinEdges',edges,'Normalization','cdf','DisplayStyle','stairs','EdgeColor','b');
    h4 = histogram(margin(4,:,i),500,'BinEdges',edges,'Normalization','cdf','DisplayStyle','stairs','EdgeColor','m');
    h5 = histogram(margin(5,:,i),500,'BinEdges',edges,'Normalization','cdf','DisplayStyle','stairs','EdgeColor','k');
    title(sprintf('Digit %d', i-1)); xlabel('Margin value'); ylabel('Cumulative Dist.');
end   
legend('t=5','t=10','t=50','t=100','t=250');

figure()
for i = 1:10
    subplot(2,5,i);
    hold on
    plot(highest_weights(:,i)); 
    title(sprintf('Digit %d', i-1)); xlabel('Iteration t'); ylabel('Indx of max weight');
    xticks(0:50:250)
end 

figure()
for i = 1:10
    for j = 1:3
        x = highest_weights(:,i);
        u = unique(x);
        [n,bin] = hist(x,u);
        [~,indices] = sort(-n);
        hardest_indices = u(indices(1:3));
        
        loop_image = [];
        for k = 1:3
            loop_image = [loop_image; reshape(train_imgs(hardest_indices(k),:),[28 28])];
        end
        subplot(5,2,i), imshow(loop_image')
        title(sprintf('Digit %d', i-1));
    end
end

max_iter = 250;
figure();
for class = 1:10
    Threshold_matrix = ones(28,28)*128;
    for i = 1:max_iter
        threshold = at_save(class,i,1);
        dim = at_save(class,i,2);
        sign = (threshold <=51);
        row = floor(dim/28) + 1;
        column = rem(dim,28);
        if column == 0
            column = 28;
        end
        Threshold_matrix (row,column) = sign*255;
    end
    
    subplot(2,5,class), imshow(Threshold_matrix, [0 255])
    title(sprintf('Digit %d', class-1));
end