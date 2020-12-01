clc
clear all
trainset = 'trainset\subset';
testset = 'testset\subset';

TrainSet = [];
TestSet = [];

for i = 0:5
    current = strcat(trainset,int2str(i));
    jpgfiles = dir(fullfile(current,'\*.jpg*'));
    n = numel(jpgfiles);
    
    for j = 1:40
        im = jpgfiles(j).name;
        im1 = imread(fullfile(current,im));
        [irow, icol] = size(im1);
        temp = reshape(im1, irow*icol, 1);
        TrainSet = [TrainSet temp];
        %figure()
        %imshow(im1);
    end
end

for i = 6:11
    current = strcat(testset,int2str(i));
    jpgfiles = dir(fullfile(current,'\*.jpg*'));
    n = numel(jpgfiles);
    
    for j = 1:10
        im = jpgfiles(j).name;
        im1 = imread(fullfile(current,im));
        [irow, icol] = size(im1);
        temp = reshape(im1, irow*icol, 1);
        TestSet = [TestSet temp];
        %figure()
        %imshow(im1);
    end
end

TrainSet = im2double(TrainSet);
TestSet = im2double(TestSet);
m = mean(TrainSet,2); 
Train_Number = size(TrainSet,2);

A = [ ];  

for i = 1 : Train_Number
    temp = double(TrainSet(:,i)) - m; 
    A = [A temp];
end

L = A'*A; 

[V, D] = eig(L); 
D_vector = diag(D);
[values, indices] = sort(D_vector,'descend');

Eigenfaces_PCA = zeros(2500,30);
for i = 1 : 30
    Eigenfaces_PCA(:,i) = A * V(:,indices(i));
end

figure()
for i = 1:16
    eigenface = Eigenfaces_PCA(:, i);
    eigenface = reshape(eigenface,50,50);
    subplot(4,4,i);
    min1 = min(min(eigenface));
    max1 = max(max(eigenface));
    eigenface=((eigenface-min1).*1)./(max1-min1);
    imshow(eigenface)
end

count = 0;
Eigenfaces_LDA = zeros(2500,15);
combs = combnk([0 1 2 3 4 5],2);

for k = 1: size(combs,1)
    combin = combs(k,:);
    i = combin(1); j = combin(2);
    count = count + 1;
    class0 = TrainSet(:,i*40 + 1:(i+1)*40);
    class1 = TrainSet(:,j*40 + 1:(j+1)*40);

    mu0 = mean(class0,2);
    mu1 = mean(class1,2);
    SB = (mu1 - mu0) * (mu1 - mu0)';

    E0 = cov(class0');
    E1 = cov(class1');
    SW = E0 + E1 + eye(size(E0,1));

    %LDA = inv(SW)*SB;
    LDA = SW\SB;
    [vectors, values] = eig(LDA);
    vectors = real(vectors);
    values = real(values);
    values_vector = diag(values);
    [values, indices] = sort(values_vector, 'descend');

    Eigenfaces_LDA(:,count) = vectors(:,indices(1));

end

figure()
for i = 1:15
    eigenface = Eigenfaces_LDA(:, i);
    eigenface = reshape(eigenface,50,50);
    subplot(4,4,i);
    min1 = min(min(eigenface));
    max1 = max(max(eigenface));
    eigenface=((eigenface-min1).*1)./(max1-min1);
    imshow(eigenface)
end


% Test Phase, Part C

Eigenfaces_PCA_Test = Eigenfaces_PCA(:,1:15)';
Z_Values_PCA = Eigenfaces_PCA_Test * TrainSet;

mu_values = zeros(15,6);
cov_values = cell(6);
for i = 1:6
    mu_values(:,i) = mean(Z_Values_PCA(:,(i-1)*40 + 1:i*40),2);
    cov_values{i} = cov(Z_Values_PCA(:,(i-1)*40 + 1:i*40)') + eye(15)*0.85;
end

TestP_PCA = Eigenfaces_PCA_Test * TestSet;

id_errors = zeros(6,1);
for i = 1:size(TestP_PCA,2)
    point = TestP_PCA(:,i);
    y = zeros(6,1);
    
    for j = 1:6
        y(j) = mvnpdf(point',mu_values(:,j)',cov_values{j});
    end
    
    [M,indices] = sort(y,'descend');
    class_id = indices(1);
    class_id = class_id - 1;
    real_id = floor(i / 10);
    if real_id == 6
        real_id = 5;
    end
    error = (class_id ~= real_id);
    id_errors(real_id + 1) = id_errors(real_id + 1) + error;
end
disp('Q5 Part C PCA error rates, from person 1 to 6, in percent')
id_errors = ((id_errors / 10)')*100;
disp(id_errors);
disp('Average error in percent:')
disp(mean(id_errors));
disp(' ');

% Test Phase, Part D

Eigenfaces_LDA_Test = Eigenfaces_LDA';
Z_Values_LDA = Eigenfaces_LDA_Test * TrainSet;

mu_values = zeros(15,6);
cov_values = cell(6);
for i = 1:6
    mu_values(:,i) = mean(Z_Values_LDA(:,(i-1)*40 + 1:i*40),2);
    cov_values{i} = cov(Z_Values_LDA(:,(i-1)*40 + 1:i*40)');
end

TestP_LDA = Eigenfaces_LDA_Test * TestSet;

id_errors = zeros(6,1);
for i = 1:size(TestP_LDA,2)
    point = TestP_LDA(:,i);
    y = zeros(6,1);
    
    for j = 1:6
        y(j) = mvnpdf(point',mu_values(:,j)',cov_values{j});
    end
    
    [M,indices] = sort(y,'descend');
    class_id = indices(1);
    class_id = class_id - 1;
    real_id = floor(i / 10);
    if real_id == 6
        real_id = 5;
    end
    error = (class_id ~= real_id);
    id_errors(real_id + 1) = id_errors(real_id + 1) + error;
end

disp('Q5 Part D LDA error rates, from person 1 to 6, in percent')
id_errors = ((id_errors / 10)')*100;
disp(id_errors);
disp('Average error in percent:')
disp(mean(id_errors));
disp(' ');

% Test Phase, Part E

TrainSet_PartE = Eigenfaces_PCA' * TrainSet;

count = 0;
combs = combnk([0 1 2 3 4 5],2);
Eigenfaces_PCA_LDA = zeros(30, 15);
for k = 1: size(combs,1)
    combin = combs(k,:);
    i = combin(1); j = combin(2);
    count = count + 1;
    class0 = TrainSet_PartE(:,i*40 + 1:(i+1)*40);
    class1 = TrainSet_PartE(:,j*40 + 1:(j+1)*40);

    mu0 = mean(class0,2);
    mu1 = mean(class1,2);
    SB = (mu1 - mu0) * (mu1 - mu0)';

    E0 = cov(class0');
    E1 = cov(class1');
    SW = E0 + E1;

    %LDA = inv(SW)*SB;
    LDA = SW\SB;
    [vectors, values] = eig(LDA);
    vectors = real(vectors);
    values = real(values);
    values_vector = diag(values);
    [values, indices] = sort(values_vector, 'descend');

    Eigenfaces_PCA_LDA(:,count) = vectors(:,indices(1));

end

TrainSet_PartE_LDA = Eigenfaces_PCA_LDA' * TrainSet_PartE;

mu_values = zeros(15,6);
cov_values = cell(6);
for i = 1:6
    mu_values(:,i) = mean(TrainSet_PartE_LDA(:,(i-1)*40 + 1:i*40),2);
    cov_values{i} = cov(TrainSet_PartE_LDA(:,(i-1)*40 + 1:i*40)');
end

TestP_PCA_LDA = Eigenfaces_PCA_LDA' * (Eigenfaces_PCA' * TestSet);

id_errors = zeros(6,1);
for i = 1:size(TestP_PCA_LDA,2)
    point = TestP_PCA_LDA(:,i);
    y = zeros(6,1);
    
    for j = 1:6
        y(j) = mvnpdf(point',mu_values(:,j)',cov_values{j});
    end
    
    [M,indices] = sort(y,'descend');
    class_id = indices(1);
    class_id = class_id - 1;
    real_id = floor(i / 10);
    if real_id == 6
        real_id = 5;
    end
    error = (class_id ~= real_id);
    id_errors(real_id + 1) = id_errors(real_id + 1) + error;
end

disp('Q5 Part E PCA + LDA error rates, from person 1 to 6, in percent:')
id_errors = ((id_errors / 10)')*100;
disp(id_errors);
disp('Average error in percent:')
disp(mean(id_errors));
disp(' ');
