clc;
clear all;
% trainImg = 'train-images.idx3-ubyte';
% trainLabel = 'train-labels.idx1-ubyte';
% 
% testImg = 't10k-images.idx3-ubyte';
% testLabel = 't10k-labels.idx1-ubyte';
% 
% [train_imgs, train_labels] = readMNIST(trainImg, trainLabel, 20000, 0);
% [test_imgs, test_labels] = readMNIST(testImg, testLabel, 10000, 0);
% 
% save HW4_Data

load HW4_Data

accuracies = zeros(3,10);
sv_counts = zeros(3,10);
% 3 different C values
margins_linear = zeros(30,20000);

loop = 0;
C_count = 0;
tic
for C = [2,4,8]
    C_count = C_count + 1;
    figure()
    sgtitle(sprintf('Results with C = %d',C));
    [ha, ~] =  tight_subplot(10,2,[0.04 0.001],[.01 .1],[.01 .01]);
    inner_loop = -1;
    for classifier = 1:10     % 10 different classifiers
        loop = loop + 1
        inner_loop = inner_loop + 2;
        tr_labels = train_labels;
        tr_labels(tr_labels ~= classifier - 1) = -1;
        tr_labels(tr_labels ~= -1) = 1;

        %Linear SVM
        svm_model = svmtrain(tr_labels, sparse(train_imgs), sprintf('-q -t 0 -c %d',C));

        te_labels = test_labels;
        te_labels(te_labels ~= classifier - 1) = -1;
        te_labels(te_labels ~= -1) = 1;

        [~, accuracy, ~] = svmpredict(te_labels, sparse(test_imgs), svm_model, '-q');
        sv_counts(C_count, classifier) = svm_model.totalSV;
        accuracies(C_count, classifier) = accuracy(1);

        [coefficients, indices] = sort(svm_model.sv_coef);
        svms = svm_model.sv_indices;
        negative_coefs = svms(indices(1:3));
        positive_coefs = svms(indices(size(indices)-2 : size(indices)));

        loop_image1 = [];
        loop_image2 = [];
        
        for i = 1:3
            loop_image1 = [loop_image1; reshape(train_imgs(negative_coefs(i),:),[28 28])];
            loop_image2 = [loop_image2; reshape(train_imgs(positive_coefs(i),:),[28 28])];
        end
        
%        axes(ha(inner_loop)); imshow(loop_image1');
%        title(sprintf('Digit %d, y = -1',classifier-1));
%        axes(ha(inner_loop+1)); imshow(loop_image2');
%        title(sprintf('Digit %d, y = 1',classifier-1));
        
        w = (svm_model.sv_coef)'*(svm_model.SVs);
        b = svm_model.rho;
        margins_linear(loop,:) = tr_labels' .* (w*(train_imgs') + b);
    end 
    
%     svm_model = svmtrain(train_labels, sparse(train_imgs), sprintf('-q -t 0 -c %d',C));
%     [~, accuracy, ~] = svmpredict(test_labels, sparse(test_imgs), svm_model, '-q');
end
toc

save HW4_Linear_Results
load HW4_Linear_Results

no_points = 1000;
loop = 0;
for C = [2,4,8]
    figure();
    sgtitle(sprintf('Results with C = %d',C));
    for i = 1:10
        index = loop*10 + i;
        lower = min(margins_linear(index,:))-3;
        upper = max(margins_linear(index,:))+3;
        edges = linspace(lower,upper,no_points);
        subplot(4,3,i);
        h1 = histogram(margins_linear(index,:),no_points,'BinEdges',edges,'Normalization','cdf','DisplayStyle','stairs','EdgeColor','r');
        xlim([lower upper-0.01])
        title(sprintf('Digit %d Margin CDF',i-1)); xlabel('Margin'); ylabel('CDF');
    end
    loop = loop + 1;
end
% [ha, pos] = tight_subplot(3,2,[.01 .03],[.1 .01],[.01 .01]) 
% for ii = 1:6; axes(ha(ii)); plot(randn(10,ii)); end 
% set(ha(1:4),'XTickLabel',''); set(ha,'YTickLabel','')

libsvmwrite('HW4_Train_Data', train_labels, sparse(train_imgs));

data_size = 20000;
sample_size = 1000;
random_indices = randperm(data_size,sample_size);
%random_indices = sort(random_indices);
libsvmwrite('HW4_Train_Data_Subset', train_labels(random_indices), sparse(train_imgs(random_indices,:)));

% grid.py used to determine C, y
% Anaconda prompt
% cd C:\Users\Asus\Desktop\WI19 Courses\ECE 271B\MATLAB\MATLAB_HW4
% python grid.py -svmtrain "C:\Users\Asus\Desktop\WI19 Courses\ECE 271B\MATLAB\libsvm-3.23\libsvm-3.23\windows\svm-train.exe" -gnuplot "C:\Program Files\gnuplot\bin\gnuplot.exe" -log2c 0,4,1 -log2g -8,-3,1 -v 5 -m 300 HW4_Train_Data_Subset
% Command:
% Train Subset size 200:  Output: C=2.0, gamma=0.0625,   Accuracy=54.0
% Train Subset size 1000: Output: C=4.0, gamma=0.015625, Accuracy=92.7
% Values to be chosen C = 4, gamma = 0.015625

C = 4; gamma = 0.015626;

margins_rbf = zeros(30,20000);

accuracies_rbf = zeros(10,1);
sv_counts_rbf = zeros(10,1);

figure()
sgtitle(sprintf('RBF Kernel, C = %d, gamma = %f',C,gamma));
[ha, pos] =  tight_subplot(10,2,[0.06 0.001],[.01 .1],[.01 .01]);
inner_loop = -1;
tic
for classifier = 1:10
    % 10 different classifiers
        classifier
        inner_loop = inner_loop + 2;
        tr_labels = train_labels;
        tr_labels(tr_labels ~= classifier - 1) = -1;
        tr_labels(tr_labels ~= -1) = 1;

        %Linear SVM
        svm_model = svmtrain(tr_labels, sparse(train_imgs), sprintf('-q -t 2 -c %d -g %f',C,gamma));

        te_labels = test_labels;
        te_labels(te_labels ~= classifier - 1) = -1;
        te_labels(te_labels ~= -1) = 1;

        [~, accuracy, ~] = svmpredict(te_labels, sparse(test_imgs), svm_model, '-q');
        sv_counts_rbf(classifier) = svm_model.totalSV;
        accuracies_rbf(classifier) = accuracy(1);

        [coefficients, indices] = sort(svm_model.sv_coef);
        svms = svm_model.sv_indices;
        negative_coefs = svms(indices(1:3));
        positive_coefs = svms(indices(size(indices)-2 : size(indices)));

        loop_image1 = [];
        loop_image2 = [];
        
        for i = 1:3
            loop_image1 = [loop_image1; reshape(train_imgs(negative_coefs(i),:),[28 28])];
            loop_image2 = [loop_image2; reshape(train_imgs(positive_coefs(i),:),[28 28])];
        end
        
        axes(ha(inner_loop)); imshow(loop_image1');
        title(sprintf('Digit %d, y = -1',classifier-1));
        axes(ha(inner_loop+1)); imshow(loop_image2');
        title(sprintf('Digit %d, y = 1',classifier-1));
        
        w = (svm_model.sv_coef)'*(svm_model.SVs);
        b = svm_model.rho;
       
        margins_rbf(classifier,:) = tr_labels' .* (w*(train_imgs') + b);
end 
%     svm_model = svmtrain(train_labels, sparse(train_imgs), sprintf('-q -t 0 -c %d',C));
%     [~, accuracy, ~] = svmpredict(test_labels, sparse(test_imgs), svm_model, '-q');
toc

save HW4_RBF_Results
load HW4_RBF_Results

no_points = 1000;
loop = 0;
figure();
C = 4;
sgtitle(sprintf('RBF Kernel, C = %d, gamma = %f',C,gamma));
for i = 1:10
    index = loop*10 + i;
    lower = min(margins_rbf(index,:))-3;
    upper = max(margins_rbf(index,:))+3;
    edges = linspace(lower,upper,no_points);
    subplot(4,3,i);
    h1 = histogram(margins_rbf(index,:),no_points,'BinEdges',edges,'Normalization','cdf','DisplayStyle','stairs','EdgeColor','r');
    xlim([lower upper-0.01])
    title(sprintf('Digit %d Margin CDF',i-1)); xlabel('Margin'); ylabel('CDF');
end
loop = loop + 1;
