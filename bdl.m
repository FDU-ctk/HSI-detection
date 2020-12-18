function B = bdl(Y, T, gamma, m, sigma)
% background dictionary learning based on locality-based linear coding

% sigma is used to diminish the step length eta as eta = sigma*eta; 

[b, n] = size(Y);
[~, c] = size(T);
Y1 = normc(Y);

T = normc(T);

B = rand(b, m); % initial bac dic
B = normc(B);

eta = 5;
for i = 1:n
    put = Y1(:,i); % pixel under test
    D = [B, T]; % union dictionary
    weight1 = sqrt(sum((repmat(put,1,m + c) - D).^2)); % locality adaptor
    
    code = (D'*D + gamma*diag(weight1.^2))\(D'*put); % LLC
    
    descent = -2*(put - D*code)*(code(1:m))';
        
    B = B - eta*descent;
    B = max(0,B);
    B = normc(B);
    
    if mod(i,50)==0
        eta = eta*sigma;
%         pause(0.05);plot(B);title(['iter=' num2str(i)])
    end
end
