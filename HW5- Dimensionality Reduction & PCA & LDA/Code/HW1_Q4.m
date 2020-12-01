clc
clear all

muA = [10; 0]; sigA = [1 0;0 2];
muB = [2; 0]; sigB = [1 0;0 10];

mu1A = muA; mu2A = -muA;
mu1B = muB; mu2B = -muB;

%For each class, 500 points
cases = 500;

%r = mvnrnd(MU,SIGMA,cases)
res1A = mvnrnd(mu1A,sigA,cases);
res2A = mvnrnd(mu2A,sigA,cases);

res1B = mvnrnd(mu1B,sigB,cases);
res2B = mvnrnd(mu2B,sigB,cases);

resB = mvnrnd(muB,sigB,cases);

% Drawing the distributions
figure();
subplot(1,2,1);
scatter(res1A(:,1),res1A(:,2),'b');
hold on
scatter(res2A(:,1),res2A(:,2),'r');
title('Q4 b) Condition A')
xlim([-15 15]);
ylim([-10 10]);
hold off

subplot(1,2,2);
scatter(res1B(:,1),res1B(:,2),'b');
hold on
scatter(res2B(:,1),res2B(:,2),'r');
xlim([-15 15]);
ylim([-10 10]);
title('Q4 b) Condition B')
hold off

% Creating two different datasets from the 2 conditions
CondA = [res1A; res2A];
CondB = [res1B; res2B];

% Sample Means & Covariances
MuA = mean(CondA); CovA = cov(CondA);
MuB = mean(CondB); CovB = cov(CondB);

% PCA Section
[egVec_A, egVal_A] = eig(CovA);
[value, index] = max(max(egVal_A));
compA = egVec_A(:,index);
[egVec_B, egVal_B] = eig(CovB);
[value, index] = max(max(egVal_B));
compB = egVec_B(:,index);

drawArrow = @(x,y) quiver( x(1),y(1),x(2)-x(1),y(2)-y(1),0, 'k', 'LineWidth',3) ;

% Drawing PCA Components

figure();
subplot(1,2,1);
scatter(res1A(:,1),res1A(:,2),'b');
hold on
scatter(res2A(:,1),res2A(:,2),'r');
%Draw function a bit funky, I will print compA for clarification
compA'
drawArrow(6*([0;0]' - compA') - [6 0], [0;0]);
title('Q4 c) Condition A')
xlim([-15 15]);
ylim([-10 10]);

subplot(1,2,2);
scatter(res1B(:,1),res1B(:,2),'b');
hold on
scatter(res2B(:,1),res2B(:,2),'r');
%Draw function a bit funky, I will print compB for clarification
compB'
drawArrow([0;0], 6*([0;0] - compB));
title('Q4 c) Condition B')
xlim([-15 15]);
ylim([-10 10]);


% Sample Means & Covariances
Mu1A = mean(res1A); Cov1A = cov(res1A);
Mu2A = mean(res2A); Cov2A = cov(res2A);
Mu1B = mean(res1B); Cov1B = cov(res1B);
Mu2B = mean(res2B); Cov2B = cov(res2B);

% LDA with regularization

SWA = Cov1A + Cov2A + eye(size(Cov1A,1));
SWB = Cov1B + Cov2B + eye(size(Cov1A,1));
SBA = (Mu2A' - Mu1A') * (Mu2A' - Mu1A')';
SBB = (Mu2B' - Mu1B') * (Mu2B' - Mu1B')';
LD_A = inv(SWA) * SBA;
LD_B = inv(SWB) * SBB;

[egVec_A_LD, egVal_A_LD] = eig(LD_A);
[value, index] = max(max(egVal_A_LD));
compA_LD = egVec_A_LD(:,index);

[egVec_B_LD, egVal_B_LD] = eig(LD_B);
[value, index] = max(max(egVal_B_LD));
compB_LD = egVec_B_LD(:,index);

% Drawing LDA Components

figure();
subplot(1,2,1);
scatter(res1A(:,1),res1A(:,2),'b');
hold on
scatter(res2A(:,1),res2A(:,2),'r');
%Draw function a bit funky, I will print the component for clarification
compA_LD'
drawArrow(6*([0;0]' - compA_LD') + [6 0], [0;0]);
title('Q4 d) Condition A')
xlim([-15 15]);
ylim([-10 10]);

subplot(1,2,2);
scatter(res1B(:,1),res1B(:,2),'b');
hold on
scatter(res2B(:,1),res2B(:,2),'r');
%Draw function a bit funky, I will print the component for clarification
compB_LD'
drawArrow(4*([0;0]' - compB_LD') + [4 0], [0;0]);
title('Q4 d) Condition B')
xlim([-15 15]);
ylim([-10 10]);