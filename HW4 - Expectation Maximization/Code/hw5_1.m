% ****** HW5 Part 1 ******* %

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

dim = 64;
c = 8;

%For test purposes
p_size = 5;
%p_size = 1;

priors_FG = zeros(5,c);
means_FG = cell(5,1);
covs_FG = cell(5,1);

priors_BG = zeros(5,c);
means_BG = cell(5,1);
covs_BG = cell(5,1);


%FG class: Random initialization/kmeans and EM algorythm for 5 different
%initializations
for num = 1:p_size
    num
    
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
        
    priors_FG(num, :) = cur_prior;
    means_FG{num} = cur_mean;
    covs_FG{num} = var;
end

%BG class: Random initialization/kmeans and EM algorythm for 5 different
%initializations
for num = 1:p_size
    num
    
    %Prior initialization
    start_prior = (ones(1,c));
    start_prior = start_prior / c;
    
    %Mean initialization by kmeans
    [labels, start_mean] = kmeans(BG, c);
    clear labels
    
    start_cov = zeros(c,dim);
    
    %Random covariance initialization
    cov_diag = rand(c,dim); 
    cov_diag(cov_diag < 0.0005) = 0.0005;

    for component = 1 : c
        start_cov(component,:) = cov_diag(component, :);
    end

    %EM algorhtym implemented in seperate .m file
    [cur_mean, var, cur_prior] = EM(BG, c, start_mean, start_cov, start_prior);

    priors_BG(num, :) = cur_prior;
    means_BG{num} = cur_mean;
    covs_BG{num} = var;
end

%This matrix will be used to store all error rates 11 x 5 x 5
error_matrix = zeros(11,5,5);

%For test purposes
dim = 64;
dim = 2;

dim_count = 0;

%Saving variables for repeated test cases
save('hw5_part1_variables.mat') 

% Main loop to try decision function for each of the 25 mixtures, 11
% idmensions
for dim = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
 
        dim_count = dim_count + 1
        cur_cov_BG = cell(5);
        cur_mean_BG = cell(5);
        cur_cov_FG = cell(5);
        cur_mean_FG = cell(5);
        
        % Getting the relevant stored variables from the above part for
        % this iteration of the loop
        for mixture = 1:p_size
            m1 = covs_BG{mixture};
            m2 = means_BG{mixture};
            m3 = covs_FG{mixture};
            m4 = means_FG{mixture};
            cur_cov_BG{mixture} = m1(:, 1:dim);
            cur_mean_BG{mixture} = m2(:, 1:dim);
            cur_cov_FG{mixture} = m3(:, 1:dim);
            cur_mean_FG{mixture} = m4(:, 1:dim);
        end
   
        % Trying 5x5 each feature in this loop
        for mixFG = 1:5
            
            for mixBG = 1:5

                count = 1;
                decisionImage = zeros(size(cImageOld,1),size(cImageOld,2));
                
                % Pixel class decision is done in the below loop
                for i = 1:cImageOldX
                    
                    for j = 1:cImageOldY  
                        
                        point = points(count,1:dim);
                        
                        %Calculations for the background probability
                        
                        tot_prob_BG = 0;
                        
                        cur_mean = cur_mean_BG{mixBG};
                        cur_sig = cur_cov_BG{mixBG};
                        cur_prior = priors_BG(mixBG,:);
                        
                        %Summing each of the hidden class' likelihoods
                        for component = 1 : c
                            tot_prob_BG = tot_prob_BG + mvnpdf(point, cur_mean(component, :), diag(cur_sig(component,:))) *  cur_prior(component);
                        end
                        
                        %Calculations for the foreground probability
                        
                        tot_prob_FG = 0;
                        
                        cur_mean = cur_mean_FG{mixFG};
                        cur_sig = cur_cov_FG{mixFG};
                        cur_prior = priors_FG(mixFG,:);
                        
                        %Summing each of the hidden class' likelihoods
                        for component = 1 : c
                            tot_prob_FG = tot_prob_FG + mvnpdf(point, cur_mean(component, :), diag(cur_sig(component,:))) *  cur_prior(component);
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
                error_matrix(dim_count, mixBG, mixFG) = CPrior * beta_error + NCPrior * alpha_error;
                
%                 figure()
%                 I = mat2gray(decisionImage,[0 1]);
%                 imshow(I); title(sprintf('Prediction image'));
                
            end
        end    
end

%Drawing the plots

for i = 1:5
    
    figure();
    x = [1:11];
    err1 = error_matrix(:, i, 1);
    err2 = error_matrix(:, i, 2);
    err3 = error_matrix(:, i, 3);
    err4 = error_matrix(:, i, 4);
    err5 = error_matrix(:, i, 5);
    
    hold on
    p1 = plot(x, err1, 'r'); L1 = "FG initizalization 1";
    p2 = plot(x, err2, 'b'); L2 = "FG initizalization 2";
    p3 = plot(x, err3, 'g'); L3 = "FG initizalization 3";
    p4 = plot(x, err4, 'm'); L4 = "FG initizalization 4";
    p5 = plot(x, err5, 'k'); L5 = "FG initizalization 5";
    lgd = legend([p1,p2,p3,p4,p5], [L1, L2, L3, L4, L5]);
    lgd.Position = [0.75 0.8 0 0];
    title(sprintf('BG initialization %d',i));
    xlabel('Dimensions'); ylabel('Probability of Error');
    hold off
    
end      