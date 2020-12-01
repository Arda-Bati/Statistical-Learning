clc
clear all
trainset = 'trainset\subset';
testset = 'testset\subset';

T = [];

for i = 0:0
    current = strcat(trainset,int2str(i));
    jpgfiles = dir(fullfile(current,'\*.jpg*'));
    n = numel(jpgfiles);
    
    for j = 1:40
        im = jpgfiles(j).name;
        im1 = imread(fullfile(current,im));
        [irow, icol] = size(im1);
        temp = reshape(im1', 1, []);
        T = [T; temp];
        %figure()
        %imshow(im1);
    end
end

T = im2double(T);

one_matx = ones(40,40);
diagonal = eye(40);
XCT = (diagonal - (1/40)*(one_matx))*T;
[M,PI,N] = svd(XCT);

PI_diag = PI(1:40,1:40);
PI_diag = diag(PI_diag);

first = N(:,1);
max_value = max(max(first));
first = (first / max_value);

first_alt = N(1,:);
second = N(:,2);


imshow(reshape(first,50,50)')
imshow(reshape(second,50,50)')

% Columns of n are eigenvectors
