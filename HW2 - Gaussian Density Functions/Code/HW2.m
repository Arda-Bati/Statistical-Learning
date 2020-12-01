%**********************
% Arda Cankat Bati
% ECE 271A - Homework#2
%***********************

clear
clc
load('TrainingSamplesDCT_8_new.mat');

%*********** GETTING THE DATA AND ESTIMATING PRIORS **********

FG = TrainsampleDCT_FG;
BG = TrainsampleDCT_BG;
FG_size = size(FG,1);
BG_size = size(BG,1);
sampleSize = FG_size + BG_size;

sampleCount = zeros(sampleSize,1);
sampleCount(1:FG_size) = 1;
hist = histogram(sampleCount);
arg = (hist.Values)/(sampleSize);
bar([0 1], arg,0.2)
xlim([-0.5 1.5]);
title('Histogram estimation, x = 0 is BG, x = 1 is FG')
xlabel('X'); ylabel('Histogram count of the class FG or BG')

CPrior = FG_size/(sampleSize);
fprintf('"Cheetah prior" estimated from the training data is: %d\n',CPrior);
NCPrior = BG_size/(sampleSize);
fprintf('"No Cheetah prior" estimated from the training data is: %d\n',NCPrior);

%********************************************************


%******** SAMPLE MEAN & VARIANCE CALCULATION***********

% Sample Mean calculation
sampleMeanFG = mean(FG)';
sampleMeanBG = mean(BG)';

%Covariance matrix calculation for 64 dimensional prob. distribution
CovMtxFG64 = cov(FG);
CovMtxBG64 = cov(BG);

%***************************************************


%************ DRAWING THE REQUIRED PLOTS **************

%For all 64 Features plotting the estimated marginal densities
count = 1;
for i = 1:16
    figure()
    sgtitle('All 64 Features Marginals') 
    for j = 1:4
        meanFG = sampleMeanFG(count);
        varFG = CovMtxFG64(count,count); stdFG = sqrt(varFG);
        meanBG = sampleMeanBG(count);
        varBG = CovMtxBG64(count,count); stdBG = sqrt(varBG);
        x1 = min(meanFG - 3*stdFG, meanBG - 3*stdBG);
        x2 = max(meanFG + 3*stdFG, meanBG + 3*stdBG);
        x = linspace(x1,x2,100);
        GaussianFG = (1/sqrt(2*pi*varFG))*exp(-((x-meanFG).^2)/(2*varFG));
        GaussianBG = (1/sqrt(2*pi*varBG))*exp(-((x-meanBG).^2)/(2*varBG));
        subplot(2,2,j)
        plot(x,GaussianFG, 'r-'); 
        hold on
        title(sprintf('Feature: %d, red-FG, black-BG',count))
        xlabel('X');
        ylabel('P(X|Y = i)');
        plot(x,GaussianBG, 'k-.');
        hold off
        count = count + 1;
    end
end

% By visual inspection from the graphs generated above,
% best and worst features are selected
BestFeatures = [1 18 19 25 27 32 33 40];
WorstFeatures = [3 4 5 59 60 62 63 64];

%For the best 8 Features plotting the estimated marginal densities
count = 1;
for i = 1:2
    figure()
    sgtitle('Best 8 Features Marginals') 
    for j = 1:4
            idx = BestFeatures(count);
            meanFG = sampleMeanFG(idx);
            stdFG =  sqrt(CovMtxFG64(idx,idx));
            meanBG = sampleMeanBG(idx);
            stdBG =  sqrt(CovMtxBG64(idx,idx));
            x1 = min(meanFG - 3*stdFG, meanBG - 3*stdBG);
            x2 = max(meanFG + 3*stdFG, meanBG + 3*stdBG);
            x = linspace(x1,x2,100);
            GaussianFG = (1/sqrt(2*pi*stdFG))*exp(-((x-meanFG).^2)/(2*(stdFG^2)));
            GaussianBG = (1/sqrt(2*pi*stdBG))*exp(-((x-meanBG).^2)/(2*(stdBG^2)));
            subplot(2,2,j)
            plot(x,GaussianFG, 'r-')  
            hold on
            plot(x,GaussianBG, 'k-.')  
            xlabel('X');
            ylabel('P(X|Y = i)');
            title(sprintf('Feature: %d, red-FG, black-BG',idx))
            hold off
            count = count+1;
    end
end

%For the worst 8 Features plotting the estimated marginal densities
count = 1;
for i = 1:2
    figure()
    sgtitle('Worst 8 Features Marginals') 
    for j = 1:4
        idx = WorstFeatures(count);
        meanFG = sampleMeanFG(idx);
        stdFG =  sqrt(CovMtxFG64(idx,idx));
        meanBG = sampleMeanBG(idx);
        stdBG =  sqrt(CovMtxBG64(idx,idx));
        x1 = min(meanFG - 3*stdFG, meanBG - 3*stdBG);
        x2 = max(meanFG + 3*stdFG, meanBG + 3*stdBG);
        x = linspace(x1,x2,100);
        GaussianFG = (1/sqrt(2*pi*stdFG))*exp(-((x-meanFG).^2)/(2*(stdFG^2)));
        GaussianBG = (1/sqrt(2*pi*stdBG))*exp(-((x-meanBG).^2)/(2*(stdBG^2)));
        subplot(2,2,j)
        plot(x,GaussianFG, 'r-')  
        hold on
        plot(x,GaussianBG, 'k-.') 
        xlabel('X');
        ylabel('P(X|Y = i)');
        title(sprintf('Feature: %d, red-FG, black-BG',idx))
        hold off
        count = count + 1;
    end
end
figure()


% For the best features calculating the 8x8 covariance matrix
%Covariance Matrices FG and BG for 8 best dimensions
%BestFeatures 
FG8 = FG(:,BestFeatures);
BG8 = BG(:,BestFeatures);

sampleMeanFG8 = sampleMeanFG(BestFeatures);
sampleMeanBG8 = sampleMeanBG(BestFeatures);

CovMtxFG8 = cov(FG8);
CovMtxBG8 = cov(BG8);

%******* GETTING THE TEST IMAGE AND PADDING *******

[cImageOld,colormap] = imread('cheetah.bmp');
cImage = im2double(cImageOld);
paddingType = 'replicate';
cImage = padarray(cImage,[4 4],paddingType,'pre');
cImage = padarray(cImage,[3 3],paddingType,'post');
imshow(cImage); title('Original image with symmetric padding.');
figure();

%****************************************************

%******* CLASSIFYING EACH PIXEL IN THE TEST IMAGE ********

cImageOldX = size(cImageOld,1); cImageOldY = size(cImageOld,2);
cImageX = size(cImage,1); cImageY = size(cImage,2);
A = [0  1  5  6  14  15  27  28 2  4  7  13  16  26  29  42 3  8  12  17  25  30  41  43 9  11  18  24  31  40  44  53 10  19  23  32  39  45  52  54 20  22  33  38  46  51  55  60 21  34  37  47  50  56  59  61 35  36  48  49  57  58  62  63];
A = A + 1;
decisionImage64 = zeros(size(cImageOld,1),size(cImageOld,2));

inv1 = inv(CovMtxFG64);
inv2 = inv(CovMtxBG64);
det1 = det(CovMtxFG64);
det2 = det(CovMtxBG64);

%All 64 Features
point = zeros(64,1);
for i = 1:cImageOldX
    for j = 1:cImageOldY
        temp = (dct2(cImage(i:i+7, j:j+7)))';
        vectorDct= temp(:);
        point(A) = vectorDct;
        mhbDistFG = ((point - sampleMeanFG)')*(inv1)*(point - sampleMeanFG);
        alphaFG = log(((2*pi)^64)*det1) - 2*log(CPrior);
        mhbDistBG = ((point - sampleMeanBG)')*(inv2)*(point - sampleMeanBG);
        alphaBG = log(((2*pi)^64)*det2) - 2*log(NCPrior);
        [M,decision] = min([(mhbDistBG + alphaBG) (mhbDistFG + alphaFG)]);
        decision = decision - 1;
        decisionImage64(i,j) = decision;
    end
end

inv1 = inv(CovMtxFG8);
inv2 = inv(CovMtxBG8);
det1 = det(CovMtxFG8);
det2 = det(CovMtxBG8);

% 8 Best Features
decisionImage8 = zeros(size(cImageOld,1),size(cImageOld,2));
point = zeros(8,1);
for i = 1:cImageOldX
    for j = 1:cImageOldY
        temp = (dct2(cImage(i:i+7, j:j+7)))';
        vectorDct= temp(:);
        point(A) = vectorDct;
        point = point(BestFeatures);
        mhbDistFG = ((point - sampleMeanFG8)')*(inv1)*(point - sampleMeanFG8);
        alphaFG = log(((2*pi)^64)*det1) - 2*log(CPrior);
        mhbDistBG = ((point - sampleMeanBG8)')*(inv2)*(point - sampleMeanBG8);
        alphaBG = log(((2*pi)^64)*det2) - 2*log(NCPrior);
        [M,decision] = min([(mhbDistBG + alphaBG) (mhbDistFG + alphaFG)]);
        decision = decision - 1;
        decisionImage8(i,j) = decision;
    end
end

%****************************************************

%******* PRINTING THE FINAL BLACK & WHITE IMAGES   *******

I = mat2gray(decisionImage64,[0 1]); 
imshow(I); title('64 Features prediction with W=Cheetah, B=NoCheetah');
figure()

I = mat2gray(decisionImage8,[0 1]); 
imshow(I); title('8 Best Features prediction with W=Cheetah, B=NoCheetah');

%****************************************************


%******* TOTAL ERROR CALCULATION FOR THE TWO CASES *********

[cImageReal colormap] = imread('cheetah_mask.bmp');
cImageReal = double(cImageReal)/255;
Image_Size = size(cImageReal,2)*size(cImageReal,1);
FG_Sum = sum(sum(cImageReal));
BG_Sum = Image_Size - FG_Sum;
truePriorCheetah = FG_Sum/Image_Size;
truePriorNoCheetah = 1 - truePriorCheetah;
fprintf('True Prior for Cheetah is: %d\n',truePriorCheetah);
fprintf('True Prior for No Cheetah is: %d\n\n',truePriorNoCheetah);

% 64 Gaussian Features Error
errorMaskBG = decisionImage64 - cImageReal;
%FG misclassified as BG / total true FG
beta64 = sum(sum(errorMaskBG == -1)) / FG_Sum; %False Negative --> beta
%BG misclassified as FG / total true BG
alpha64 = sum(sum(errorMaskBG == 1)) / BG_Sum;  %False Positive --> alpha
ProbOfError64 = truePriorCheetah * beta64 + truePriorNoCheetah * alpha64;
fprintf('False Positive alpha for 64 Features is: %d\n',alpha64);
fprintf('False Negative beta for 64 Features is: %d\n',beta64);
fprintf('Total Probability of Error for 64 Features is: %d\n\n',ProbOfError64);

% 8 Gaussian Features Error
errorMaskFG = decisionImage8 - cImageReal;
%FG misclassified as BG / total true FG
beta8 = sum(sum(errorMaskFG == -1)) / FG_Sum; %False Negative --> beta
%BG misclassified as FG / total true BG
alpha8 = sum(sum(errorMaskFG == 1)) / BG_Sum;  %False Positive --> alpha
ProbOfError8 = truePriorCheetah * beta8 + truePriorNoCheetah * alpha8;
fprintf('False Positive alpha for best 8 features is: %d\n',alpha8);
fprintf('False Negative beta for best 8 features is: %d\n',beta8);
fprintf('Total Probability of Error for best 8 features is: %d\n',ProbOfError8);











% ***** PRINT OUTPUT FROM THE CODE IS BELOW ****************


