clc,clear
% close all

load urban_162band
Y = double(urban_detection);
[no_lines,no_rows,no_bands] = size(Y);
Y = reshape(Y,no_lines*no_rows,no_bands)';
Y1 = Y./max(Y(:));
Y = normc(Y);

%% BDL
ind_tar = [6261 6896];
T = Y(:,ind_tar);
gamma = 20;
m = 20;
sigma = 0.95;

rng(8, 'twister');
tic
B = bdl(Y1, T, gamma, m, sigma);
toc

%% DM
lambda = 0.01;
beta = 0.1;
W = ones(size(T,2),no_lines*no_rows);
for ii = 1:size(T,2)
    W(ii,:) = sqrt(sum((repmat(T(:,ii),1,no_lines*no_rows) - Y).^2));
end

display = true;
im_size = [no_lines,no_rows];
tic
[X, Z] = dm(Y1, B, T, lambda, beta, W, im_size, display);
toc

S = T*Z;
re = reshape(sqrt(sum((S).^2)),no_lines,no_rows);
imagesc(re);axis off

[tpr,fpr,~] = roc(groundtruth(:)',re(:)');
figure;semilogx(fpr,tpr,'b-')
set(gca,'Fontsize',12)
AUC = trapz(fpr,tpr);
title(['AUC=' num2str(AUC)])
xlabel('False alarm rate')
ylabel('Detection probability')
axis([1e-4 1 0 1])
% set(gca,'XGrid','on');
