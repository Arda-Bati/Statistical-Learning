%**********************
% Arda Cankat Bati
% ECE 271A - Homework#1
%***********************

clear
clc
load('TrainingSamplesDCT_8.mat');

%******* IMAGE EXTRACTION AND PRIOR ESIMATION *******
%******* EXTRACTING FEATURE X FROM TRAIN DATA *******

FG = abs(TrainsampleDCT_FG);
BG = abs(TrainsampleDCT_BG);
FG_size = size(FG,1);
BG_size = size(BG,1);

ArraySize = size(FG,2);
SecLarFG = zeros(1,FG_size);
SecLarBG = zeros(1,BG_size);
CheetahPrior = FG_size/(FG_size + BG_size);
fprintf('"Cheetah prior" estimated from the training data is: %d\n',CheetahPrior);
NoCheetahPrior = BG_size/(FG_size + BG_size);
fprintf('"No Cheetah prior" estimated from the training data is: %d\n',NoCheetahPrior);
for n = 1:size(FG,1)
    [M,I] = max(FG(n,:));
    FG(n,I) = 0;
    [M,SecLarFG(n)] = max(FG(n,:));
end
for n = 1:size(BG,1)
    [M,I] = max(BG(n,:));
    BG(n,I) = 0;
    [M,SecLarBG(n)] = max(BG(n,:));
end

%****************************************************

%******* CREATING HISTOGRAMS FOR CONDITIONAL PROBS OF X *******

XGivenCheetah = histogram(SecLarFG,'Normalization','PDF');
X1 = XGivenCheetah.Values;
XGivenCheetah.BinLimits = [1.5 64.5];
title('Probability distribution of X for Training Data = Cheetah');
xlabel('X, index of the second largest(absolute) DCT value'); 
ylabel('P(X | Y = Cheetah)'); figure();
XGivenNoCheetah = histogram(SecLarBG,'Normalization','PDF');
X2 = XGivenNoCheetah.Values;
XGivenNoCheetah.BinLimits = [1.5 64.5];
title('Probability distribution of X for Training Data = No Cheetah');
xlabel('X, index of the second largest(absolute) DCT value'); 
ylabel('P(X | Y = No Cheetah)'); figure();
X1 = padarray(X1,[0 (ArraySize - size(X1,2))], 'post');
X2 = padarray(X2,[0 (ArraySize - size(X2,2))], 'post');

%Regularization to mitigate the empty bins
X1 = X1 + 0.005;
X1 = X1./sum(X1);
X2 = X2 + 0.001;
X2 = X2./sum(X2);
%****************************************************

%******* GETTING THE TEST IMAGE AND PADDING *******

[cImageOld colormap] = imread('cheetah.bmp');
[cImageOld,colormap] = imread('cheetah.bmp');
cImage = im2double(cImageOld);
paddingType = 'replicate';
cImage = padarray(cImage,[4 4],paddingType,'pre');
cImage = padarray(cImage,[3 3],paddingType,'post');
imshow(cImage); title('Original image with 0 padding.'); figure();

%****************************************************

%******* CLASSIFYING EACH PIXEL IN THE TEST IMAGE ********
%******* PRINTING THE FINAL BLACK & WHITE IMAGE    *******

cImageOldX = size(cImageOld,1); cImageOldY = size(cImageOld,2);
cImageX = size(cImage,1); cImageY = size(cImage,2);
A = [0  1  5  6  14  15  27  28 2  4  7  13  16  26  29  42 3  8  12  17  25  30  41  43 9  11  18  24  31  40  44  53 10  19  23  32  39  45  52  54 20  22  33  38  46  51  55  60 21  34  37  47  50  56  59  61 35  36  48  49  57  58  62  63];
A = A + 1;
decisionImage = zeros(size(cImageOld,1),size(cImageOld,2));

for i = 1:cImageOldX
    for j = 1:cImageOldY
        temp = (abs(dct2(cImage(i:i+7, j:j+7))))';
        vectorDct= temp(:);
        zigzag(A) = vectorDct;
        [M,I] = max(zigzag);
        zigzag(I) = 0;
        [M,I] = max(zigzag);
        [M,decision] = max([(log(NoCheetahPrior) + log(X2(I))) (log(CheetahPrior) + log(X1(I)))]);
        decision = decision - 1;
        decisionImage(i,j) = decision;
    end
end

I = mat2gray(decisionImage,[0 1]); 
imshow(I); title('Decision image with W=Cheetah, B=NoCheetah');

%****************************************************

%******* CALCULATING THE TOTAL ERROR *******

[cImageReal colormap] = imread('cheetah_mask.bmp');
cImageReal = double(cImageReal)/255;
errorMask = decisionImage - cImageReal;
Image_Size = size(cImageReal,2)*size(cImageReal,1);
FG_Sum = sum(sum(cImageReal));
BG_Sum = Image_Size - FG_Sum;
truePriorCheetah = FG_Sum/Image_Size;
truePriorNoCheetah = 1 - truePriorCheetah;

%FG misclassified as BG / total true FG
beta = sum(sum(errorMask == -1)) / FG_Sum; %False Negative --> beta
%BG misclassified as FG / total true BG
alpha = sum(sum(errorMask == 1)) / BG_Sum;  %False Positive --> alpha
ProbOfError = truePriorCheetah * beta + truePriorNoCheetah * alpha;
fprintf('True Prior for Cheetah is: %d\n',truePriorCheetah);
fprintf('True Prior for No Cheetah is: %d\n',truePriorNoCheetah);
fprintf('False Positive alpha is: %d\n',alpha);
fprintf('False Negative beta: %d\n',beta);
fprintf('Total Probability of Error is: %d\n',ProbOfError);

%****************************************************