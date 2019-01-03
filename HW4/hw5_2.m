% ****** HW5 Part 2 ******* %

clc
clear
load ('TrainingSamplesDCT_8_new.mat');

FG = TrainsampleDCT_FG;
BG = TrainsampleDCT_BG;
FG_size = size(FG,1);
BG_size = size(BG,1);
sampleSize = FG_size + BG_size;
CPrior = FG_size/(sampleSize);
NCPrior = BG_size/(sampleSize);

[cImageOld,colormap] = imread('cheetah.bmp');
cImage = im2double(cImageOld);
paddingType = 'replicate';
cImage = padarray(cImage,[4 4],paddingType,'pre');
cImage = padarray(cImage,[3 3],paddingType,'post');
[cImageReal colormap] = imread('cheetah_mask.bmp');

figure();
imshow(cImage); title('Original image with symmetric padding.');


cImageReal = double(cImageReal)/255;
Image_Size = size(cImageReal,2)*size(cImageReal,1);
FG_Sum = sum(sum(cImageReal));
BG_Sum = Image_Size - FG_Sum;
cImageOldX = size(cImageOld,1); cImageOldY = size(cImageOld,2);
cImageX = size(cImage,1); cImageY = size(cImage,2);
A = [0  1  5  6  14  15  27  28 2  4  7  13  16  26  29  42 3  8  12  17  25  30  41  43 9  11  18  24  31  40  44  53 10  19  23  32  39  45  52  54 20  22  33  38  46  51  55  60 21  34  37  47  50  56  59  61 35  36  48  49  57  58  62  63];
A = A + 1;

decisionImage64 = zeros(size(cImageOld,1),size(cImageOld,2));
points = zeros(cImageOldX*cImageOldY,64);
point = zeros(64,1);
count = 1;

for i = 1:cImageOldX
    for j = 1:cImageOldY
    temp = (dct2(cImage(i:i+7, j:j+7)))';
    vectorDct= temp(:);
    point(A) = vectorDct;
    points(count,:) = point';
    count = count + 1;
    end
end

%For test purposes
p_size = 6;
p_size = 2;

priors_FG = cell(6,1);
means_FG = cell(6,1);
covs_FG = cell(6,1);

priors_BG = cell(6,1);
means_BG = cell(6,1);
covs_BG = cell(6,1);

dim = 64;

component_sizes = [1, 2, 4, 8, 16, 32];

%FG class: Random initialization/kmeans and EM algorythm for 6 different
%component sizes
for num = 1:p_size
    num
    c = component_sizes(num);
    
    %Prior initialization
    start_prior = (ones(1,c));
    start_prior = start_prior / c;
    
    %Mean initialization by kmeans
    [labels, start_mean] = kmeans(FG, c);
    clear labels
    
    %Random covariance initialization
    start_cov = zeros(c,dim);

    cov_diag = rand(c,dim); 
    cov_diag(cov_diag < 0.0005) = 0.0005;

    for component = 1 : c
        start_cov(component,:) = cov_diag(component, :);
    end

    %EM algorhtym implemented in seperate .m file
    [cur_mean, var, cur_prior] = EM(FG, c, start_mean, start_cov, start_prior);
        
    priors_FG{num} = cur_prior;
    means_FG{num} = cur_mean;
    covs_FG{num} = var;
end

%BG class: Random initialization/kmeans and EM algorythm for 6 different
%component sizes
for num = 1:p_size
    num
    c = component_sizes(num);
    
    %Prior initialization
    start_prior = (ones(1,c));
    start_prior = start_prior / c;
    
    %Mean initialization by kmeans
    [labels, start_mean] = kmeans(BG, c);
    clear labels
    
    %Random covariance initialization
    start_cov = zeros(c,dim);

    cov_diag = rand(c,dim); 
    cov_diag(cov_diag < 0.0005) = 0.0005;

    for component = 1 : c
        start_cov(component,:) = cov_diag(component, :);
    end

    %EM algorhtym implemented in seperate .m file
    [cur_mean, var, cur_prior] = EM(BG, c, start_mean, start_cov, start_prior);
        
    priors_BG{num} = cur_prior;
    means_BG{num} = cur_mean;
    covs_BG{num} = var;
end

save('hw5_part2_variables.mat')

%This matrix will be used to store all error rates 11 x 5 x 5
error_matrix = zeros(6,11);

%Main loop to try 6 different component sizes for 11 dimensions

   
for component_count = 1:p_size
    
    % Getting the relevant stored variables from the above part for
    % this iteration of the loop

    c = component_sizes(component_count)
    m1 = covs_BG{component_count};
    m2 = means_BG{component_count};
    m3 = covs_FG{component_count};
    m4 = means_FG{component_count};
    
    dim_count = 0;
    for dim = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]

            cur_cov_BG = m1(:, 1:dim);
            cur_mean_BG = m2(:, 1:dim);
            cur_cov_FG = m3(:, 1:dim);
            cur_mean_FG = m4(:, 1:dim);
            
            dim_count = dim_count + 1
%             cur_cov_BG = cell(5);
%             cur_mean_BG = cell(5);
%             cur_cov_FG = cell(5);
%             cur_mean_FG = cell(5);

            count = 1;
            decisionImage = zeros(size(cImageOld,1),size(cImageOld,2));

            % Pixel class decision is done in the below loop
            for i = 1:cImageOldX

                for j = 1:cImageOldY  

                    point = points(count,1:dim);

                    %Calculations for the background probability
                    tot_prob_BG = 0;

                    %Summing each of the hidden class' likelihoods
                    for component = 1 : c
                        tot_prob_BG = tot_prob_BG + mvnpdf(point, cur_mean_BG(component, :), diag(cur_cov_BG(component,:))) *  priors_BG{component_count}(component);
                    end
                    
                    %Calculations for the foreground probability

                    tot_prob_FG = 0;

                    %Summing each of the hidden class' likelihoods
                    for component = 1 : c
                        tot_prob_FG = tot_prob_FG + mvnpdf(point, cur_mean_FG(component, :), diag(cur_cov_FG(component,:))) *  priors_FG{component_count}(component);
                    end

                    %Main decision function
                    [M,decision] = max([(tot_prob_BG * NCPrior) (tot_prob_FG * CPrior)]);
                    decision = decision - 1;
                    decisionImage(i,j) = decision;
                    count = count + 1;
                end

            end

            errorMaskBG = decisionImage - cImageReal;
            %FG misclassified as BG / total true FG
            beta_error = sum(sum(errorMaskBG == -1)) / FG_Sum; %False Negative --> beta
            %BG misclassified as FG / total true BG
            alpha_error = sum(sum(errorMaskBG == 1)) / BG_Sum;  %False Positive --> alpha
            error_matrix(component_count, dim_count) = CPrior * beta_error + NCPrior * alpha_error;

%                 figure()
%                 I = mat2gray(decisionImage,[0 1]);
%                 imshow(I); title(sprintf('Prediction image'));

    end    
end

save('hw5_part2_variables.mat')
    
figure();
x_dimension = [1:11];
err1 = error_matrix(1,:);
err2 = error_matrix(2,:);
err3 = error_matrix(3,:); %problem
err4 = error_matrix(4,:);
err5 = error_matrix(5,:); %problem
err6 = error_matrix(6,:); %problem

%component_sizes = [1, 2, 4, 8, 16, 32];
hold on
p1 = plot(x, err1, 'r'); L1 = "1 Mixture Component";
p2 = plot(x, err2, 'b'); L2 = "2 Mixture Components";
p3 = plot(x, err3, 'g'); L3 = "4 Mixture Components";
p4 = plot(x, err4, 'm'); L4 = "8 Mixture Components";
p5 = plot(x, err5, 'k'); L5 = "16 Mixture Components";
p6 = plot(x, err6, 'c'); L6 = "32 Mixture Components";
lgd = legend([p1,p2,p3,p4,p5,p6], [L1, L2, L3, L4, L5, L6]);
lgd.Position = [0.75 0.8 0.2 0.2];
title('Gaussian Mixtures, different component values');
xlabel('Dimensions'); ylabel('Probability of Error');
hold off
    