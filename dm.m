function [X, Z] = dm(Y, B, T, lambda, beta, W, im_size, display)
% decomposition model with total variation and sparsity regularization
% this routine solves the following optimization problem
% min ||Y-B*X-T*Z||_F^2+lambda*||HX||_1,1+beta*||W.*Z||_1,1

% if the observed data are of size no_lines*no_rows*no_bands, then 
% im_size = [no_lines, no_rows];

[~, n] = size(Y);
m = size(B, 2);
c = size(T, 2);
maxIter = 400;

rho = 1.2;
mu = 1e-3;% initial mu and rho, can be tuned
mu_bar = 1e10;
% initialization
X0 = 0;
atx_A = 2*(T'*T);
atx_Y = 2*(T'*Y);

atx_B = 2*(B'*B);
atx_Y1 = 2*(B'*Y);

%%
% build handlers and necessary stuff
% horizontal difference operators
FDh = zeros(im_size);
FDh(1,1) = -1;
FDh(1,end) = 1;
FDh = fft2(FDh);
FDhH = conj(FDh);

% vertical difference operator
FDv = zeros(im_size);
FDv(1,1) = -1;
FDv(end,1) = 1;
FDv = fft2(FDv);
FDvH = conj(FDv);

IL = 1./( FDhH.* FDh + FDvH.* FDv + 1);

Dh = @(x) real(ifft2(fft2(x).*FDh));
DhH = @(x) real(ifft2(fft2(x).*FDhH));

Dv = @(x) real(ifft2(fft2(x).*FDv));
DvH = @(x) real(ifft2(fft2(x).*FDvH));

%%
%---------------------------------------------
%  Initializations
%---------------------------------------------

% no intial solution supplied
if X0 == 0
    X = zeros(m, n);
end

index = 1;

% initialize V variables
V = cell(3,1);

% initialize D variables (scaled Lagrange Multipliers)
D = cell(3,1);

%  data term (always present)
V{1} = X;         % V1
D{1} = zeros(m, n);

%TV
% V2
% convert X into a cube
U_im = reshape(X',im_size(1), im_size(2),m);

% V2 create two images per band (horizontal and vertical differences)
V{index+1} = cell(m,2);
D{index+1} = cell(m,2);
for kk = 1:m
    % build V2 image planes
    V{index+1}{kk}{1} = Dh(U_im(:,:,kk));   % horizontal differences
    V{index+1}{kk}{2} = Dv(U_im(:,:,kk));   %   vertical differences
    % build D2 image planes
    D{index+1}{kk}{1} = zeros(im_size);   % horizontal differences
    D{index+1}{kk}{2} = zeros(im_size);   %   vertical differences
end
clear U_im;
V{index+2} = zeros(c, n);
D{index+2} = zeros(c, n);
Z = zeros(c, n);

%%
%---------------------------------------------
%  AL iterations - main body
%---------------------------------------------
tol = sqrt(n)*1e-5;
iter = 1;
res = inf;
while (iter <= maxIter) && (sum(abs(res)) > tol)
    
    % solve the quadratic step
    X = (atx_B + mu*eye(m))\(atx_Y1 - 2*(B'*T)*Z+ mu*(V{1} - D{1})); 
        
    % solve the Z
    Z = (atx_A + mu*eye(c))\(atx_Y - 2*(T'*B)*X+ mu*(V{3} - D{3})); 
    
    j = 1;
    nu_aux = X + D{j};
    % convert nu_aux into image planes
    nu_aux_im = reshape(nu_aux',im_size(1), im_size(2),m);
    % compute V1 in the form of image planes
    for k = 1:m
        % V1
        V1_im(:,:,k) = real(ifft2(IL.*fft2(DhH(V{j+1}{k}{1}-D{j+1}{k}{1}) ...
            +  DvH(V{j+1}{k}{2}-D{j+1}{k}{2}) +  nu_aux_im(:,:,k))));
        % V2
        aux_h = Dh(V1_im(:,:,k));
        aux_v = Dv(V1_im(:,:,k));

        V{j+1}{k}{1} = soft(aux_h + D{j+1}{k}{1}, lambda/mu);   %horizontal
        V{j+1}{k}{2} = soft(aux_v + D{j+1}{k}{2}, lambda/mu);   %vertical

        % update D2
        D{j+1}{k}{1} =  D{j+1}{k}{1} + (aux_h - V{j+1}{k}{1});
        D{j+1}{k}{2} =  D{j+1}{k}{2} + (aux_v - V{j+1}{k}{2});
    end
    % convert V1 to matrix format
    V{j} = reshape(V1_im, prod(im_size),m)';            
        
    % V3
    V{j+2} = soft(Z + D{j+2}, (beta/mu).*W);
    
    % update Lagrange multipliers    
    D{j} = D{j} + (X - V{j});
    
    D{j + 2} = D{j + 2} + (Z - V{j + 2});
    
    % compute residuals
    if mod(iter,10) == 1
        st = [];
        for j = 1:2:3
            if j == 1
                res(j) = norm(X - V{j},'fro');
                st = strcat(st,sprintf('  res(%i) = %2.6f',j,res(j) ));
            else
                res(j) = norm(Z - V{j},'fro');
                st = strcat(st,sprintf('  res(%i) = %2.6f',j,res(j) ));
            end
        end
        if display
            fprintf(strcat(sprintf('iter = %i -',iter),st,'\n'));
        end
    end

    iter = iter + 1;    
    mu = min(mu*rho, mu_bar);
end
