%**********************
% Arda Cankat Bati
% ECE 271A - Homework#2
%***********************

% BAYESIAN ESTIMATION

clear
clc
load('TrainingSamplesDCT_subsets_8.mat');
load('alpha.mat');

% ML Estimation Results from Part 2
ML_Error = zeros(1,4);

FGs = {D1_FG,D2_FG,D3_FG,D4_FG};
BGs = {D1_BG,D2_BG,D3_BG,D4_BG};

%*********** GETTING THE DATA AND ESTIMATING PRIORS **********

[cImageOld,colormap] = imread('cheetah.bmp');
cImage = im2double(cImageOld);
paddingType = 'replicate';
cImage = padarray(cImage,[4 4],paddingType,'pre');
cImage = padarray(cImage,[3 3],paddingType,'post');
% imshow(cImage); title('Original image with symmetric padding.');
% figure();

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

load('Prior_1.mat');
str1_mu0_FG = mu0_FG;
str1_mu0_BG = mu0_BG;
str1_w0 = W0;

clear mu0_FG
clear mu0_BG
clear W0

load('Prior_2.mat');
str2_mu0_FG = mu0_FG;
str2_mu0_BG = mu0_BG;

str2_w0 = W0;

mu0_FG = zeros(2,64);
mu0_BG = zeros(2,64);

mu0_FG(1,:) = str1_mu0_FG;
mu0_FG(2,:) = str2_mu0_FG;
mu0_BG(1,:) = str1_mu0_BG;
mu0_BG(2,:) = str2_mu0_BG;

%************ LOOP FOR PART 1 ********************

for loop = 1:4 
    
    FG = cell2mat(FGs(loop));
    BG = cell2mat(BGs(loop));
    FG_size = size(FG,1);
    BG_size = size(BG,1);
    sampleSize = FG_size + BG_size;

    CPrior = FG_size/(sampleSize);
    NCPrior = BG_size/(sampleSize);
    
    mean1st_FG = mean(FG(:,1));
    mean1st_BG = mean(BG(:,1));
    
    fprintf('For dataset#%d mean of the 1st foreround DCT coefficient is: %f\n',loop,mean1st_FG); 
    fprintf('For dataset#%d mean of the 1st background DCT coefficient is: %f\n',loop,mean1st_BG); 

    %******** SAMPLE MEAN & VARIANCE CALCULATION***********

    % Sample Mean calculation
    sampleMeanFG = mean(FG,1)';
    sampleMeanBG = mean(BG,1)';

    %Covariance matrix calculation for 64 dimensional prob. distribution
    CovMtxFG = cov(FG);
    CovMtxBG = cov(BG);

    cov0 = zeros(2,9,64,64);

    for i=1:9
        cov0(1,i,:,:) = diag(alpha(i)*str1_w0);
        cov0(2,i,:,:) = diag(alpha(i)*str2_w0);
    end
    
    for strategy = 1:2

        for alpha_count = 1:9
            
            %fprintf('Dataset %d, Strategy %d, Alpha(%d)\n',loop,strategy,alpha_count);

            n = FG_size;
            E0 = squeeze(cov0(strategy,alpha_count,:,:));
            E = CovMtxFG;
            mn_hat = (sampleMeanFG);
            m0 = (mu0_FG(strategy,:))';

            m1_FG = (E0/(E0 + E/n))*mn_hat + ((E/n)/(E0 + E/n))*m0;
            E1_FG = (E0/(E0 + E/n))*(E/n);

            n = BG_size;
            E0 = squeeze(cov0(strategy,alpha_count,:,:));
            E = CovMtxBG;
            mn_hat = (sampleMeanBG);
            m0 = (mu0_BG(strategy,:))';

            m1_BG = (E0/(E0 + E/n))*mn_hat + ((E/n)/(E0 + E/n))*m0;
            E1_BG = (E0/(E0 + E/n))*(E/n);

            E_FG = CovMtxFG + E1_FG;
            E_BG = CovMtxBG + E1_BG;


            %******* CLASSIFYING EACH PIXEL IN THE TEST IMAGE ********

            decisionImage64 = zeros(size(cImageOld,1),size(cImageOld,2));

            %All 64 Features
            point = zeros(64,1);
            invFG = inv(E_FG);
            detFG = det(E_FG);
            invBG = inv(E_BG);
            detBG = det(E_BG);

            count = 1;
            for i = 1:cImageOldX
                for j = 1:cImageOldY
                    point = points(count,:)';
                    mhbDistFG = ((point - m1_FG)')*(invFG)*(point - m1_FG);
                    alphaFG = log(((2*pi)^64)*detFG) - 2*log(CPrior);
                    mhbDistBG = ((point - m1_BG)')*(invBG)*(point - m1_BG);
                    alphaBG = log(((2*pi)^64)*detBG) - 2*log(NCPrior);
                    [M,decision] = min([(mhbDistBG + alphaBG) (mhbDistFG + alphaFG)]);
                    decision = decision - 1;
                    decisionImage64(i,j) = decision;
                    count = count + 1;
                end
            end

            %****************************************************

            errorMaskBG = decisionImage64 - cImageReal;
            %FG misclassified as BG / total true FG
            beta64 = sum(sum(errorMaskBG == -1)) / FG_Sum; %False Negative --> beta
            %BG misclassified as FG / total true BG
            alpha64 = sum(sum(errorMaskBG == 1)) / BG_Sum;  %False Positive --> alpha
            PE_Bayes(strategy,loop,alpha_count) = CPrior * beta64 + NCPrior * alpha64;

    %         I = mat2gray(decisionImage64,[0 1]); 
    %         imshow(I); title(sprintf('Prediction image with alpha= %f',alpha));
    %         figure()

        end
    end
end

% *********** LOOP FOR PART 2 *******************

print('');

for loop=1:4

    FG = cell2mat(FGs(loop));
    BG = cell2mat(BGs(loop));
    FG_size = size(FG,1);
    BG_size = size(BG,1);
    sampleSize = FG_size + BG_size;

    CPrior = FG_size/(sampleSize);
    NCPrior = BG_size/(sampleSize);

    %******** SAMPLE MEAN & VARIANCE CALCULATION***********

    % Sample Mean calculation
    sampleMeanFG = mean(FG,1)';
    sampleMeanBG = mean(BG,1)';

    %Covariance matrix calculation for 64 dimensional prob. distribution
    CovMtxFG = cov(FG);
    CovMtxBG = cov(BG);

    %All 64 Features
    point = zeros(64,1);
    count = 1;
    for i = 1:cImageOldX
        for j = 1:cImageOldY
            point = points(count,:)';
            mhbDistFG = ((point - sampleMeanFG)')*(inv(CovMtxFG))*(point - sampleMeanFG);
            alphaFG = log(((2*pi)^64)*det(CovMtxFG)) - 2*log(CPrior);
            mhbDistBG = ((point - sampleMeanBG)')*(inv(CovMtxBG))*(point - sampleMeanBG);
            alphaBG = log(((2*pi)^64)*det(CovMtxBG)) - 2*log(NCPrior);
            [M,decision] = min([(mhbDistBG + alphaBG) (mhbDistFG + alphaFG)]);
            decision = decision - 1;
            decisionImage64(i,j) = decision;
            count = count + 1;
        end
    end

    %******* PRINTING THE FINAL BLACK & WHITE IMAGES   *******

%     I = mat2gray(decisionImage64,[0 1]); 
%     imshow(I); title('64 Features prediction with W=Cheetah, B=NoCheetah');
%     figure()

    %******* TOTAL ERROR CALCULATION FOR THE TWO CASES *********

    % 64 Gaussian Features Error
    errorMaskBG = decisionImage64 - cImageReal;
    %FG misclassified as BG / total true FG
    beta64 = sum(sum(errorMaskBG == -1)) / FG_Sum; %False Negative --> beta
    %BG misclassified as FG / total true BG
    alpha64 = sum(sum(errorMaskBG == 1)) / BG_Sum;  %False Positive --> alpha
    ProbOfError64 = CPrior * beta64 + NCPrior * alpha64;
    fprintf('ML estimation error probability for dataset#%d is: %f\n',loop,ProbOfError64);
    ML_Error(loop) = ProbOfError64;

end


% ************ LOOP FOR PART 3 ********************

for strategy = 1:2
    
    for loop = 1:4 

        FG = cell2mat(FGs(loop));
        BG = cell2mat(BGs(loop));
        FG_size = size(FG,1);
        BG_size = size(BG,1);
        sampleSize = FG_size + BG_size;

        CPrior = FG_size/(sampleSize);
        NCPrior = BG_size/(sampleSize);


        %******** SAMPLE MEAN & VARIANCE CALCULATION***********

        % Sample Mean calculation
        sampleMeanFG = mean(FG,1)';
        sampleMeanBG = mean(BG,1)';

        %Covariance matrix calculation for 64 dimensional prob. distribution
        CovMtxFG = cov(FG);
        CovMtxBG = cov(BG);
        
        cov0 = zeros(2,9,64,64);

        for i=1:9
            cov0(1,i,:,:) = diag(alpha(i)*str1_w0);
            cov0(2,i,:,:) = diag(alpha(i)*str2_w0);
        end

        for alpha_count = 1:9

            %fprintf('Dataset %d, Strategy %d, Alpha(%d)\n',loop,strategy,alpha_count);
            
            n = FG_size;
            E0 = squeeze(cov0(strategy,alpha_count,:,:));
            E = CovMtxFG;
            mn_hat = (sampleMeanFG);
            m0 = (mu0_FG(strategy,:))';

            m1_FG = (E0/(E0 + E/n))*mn_hat + ((E/n)/(E0 + E/n))*m0;
          
            n = BG_size;
            E0 = squeeze(cov0(strategy,alpha_count,:,:));
            E = CovMtxBG;
            mn_hat = (sampleMeanBG);
            m0 = (mu0_BG(strategy,:))';

            m1_BG = (E0/(E0 + E/n))*mn_hat + ((E/n)/(E0 + E/n))*m0;

            E_FG = CovMtxFG;
            E_BG = CovMtxBG;

            %******* CLASSIFYING EACH PIXEL IN THE TEST IMAGE ********

            decisionImage64 = zeros(size(cImageOld,1),size(cImageOld,2));

            %All 64 Features
            point = zeros(64,1);
            invFG = inv(E_FG);
            detFG = det(E_FG);
            invBG = inv(E_BG);
            detBG = det(E_BG);

            count = 1;
            for i = 1:cImageOldX
                for j = 1:cImageOldY
                    point = points(count,:)';
                    mhbDistFG = ((point - m1_FG)')*(invFG)*(point - m1_FG);
                    alphaFG = log(((2*pi)^64)*detFG) - 2*log(CPrior);
                    mhbDistBG = ((point - m1_BG)')*(invBG)*(point - m1_BG);
                    alphaBG = log(((2*pi)^64)*detBG) - 2*log(NCPrior);
                    [M,decision] = min([(mhbDistBG + alphaBG) (mhbDistFG + alphaFG)]);
                    decision = decision - 1;
                    decisionImage64(i,j) = decision;
                    count = count + 1;
                end
            end

            errorMaskBG = decisionImage64 - cImageReal;
            %FG misclassified as BG / total true FG
            beta64 = sum(sum(errorMaskBG == -1)) / FG_Sum; %False Negative --> beta
            %BG misclassified as FG / total true BG
            alpha64 = sum(sum(errorMaskBG == 1)) / BG_Sum;  %False Positive --> alpha
            PE_MAP(strategy,loop,alpha_count) = CPrior * beta64 + NCPrior * alpha64;

%           I = mat2gray(decisionImage64,[0 1]); 
%           imshow(I); title(sprintf('Prediction image with alpha= %f',alpha));
%           figure()
        end
    end
end


% ********* PLOTTING THE RESULTS **************

for i = 1:2
    for j = 1:4
        y1 = squeeze(PE_Bayes(i,j,:))';
        y2 = squeeze(PE_MAP(i,j,:))';
        y3(1:9) = ML_Error(j);
        hold on
        p1 = plot(alpha,y1,'ro-'); L1 = "Bayesian Estimator";
        p2 = plot(alpha,y2,'bx--'); L2 = "MAP Estimator";
        p3 = plot(alpha,y3,'k');      L3 = "ML Estimator";
        lgd = legend([p1,p2, p3], [L1, L2, L3]);
        lgd.Position = [0.75 0.5 0 0];
        hold off
        set(gca,'XScale', 'log')
        title(sprintf('Prediction Methods for, Strategy %d, Dataset# %d',i,j));
        xlabel('log(Alpha)')
        ylabel('Prob. of Error')
        figure()
    end
end

% ***** PRINT OUTPUT FROM THE CODE IS BELOW ****************


