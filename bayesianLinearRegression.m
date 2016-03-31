function [theta, llVec] = bayesianLinearRegression(x,y,iterations)

theta = [];
llVec = zeros(iterations,1);

[n, d] = size(x);

% Set the hyperparameters

varBeta = 5;
varAlpha = 5;
priorBeta = [zeros(d+1,1), varBeta*ones(d+1,1)];
priorBeta(1,2) = varAlpha;

Vbeta = diag(priorBeta(:,2));
muBeta = zeros(d+1,1);

priorSig = [1, 1];

% Make the posterior parameters
n = length(y);
xx = [ones(n,1), x];
size(Vbeta)
size(xx'*xx)
Vstar = (Vbeta^(-1) + xx'*xx)^(-1);
muStar = Vstar*(Vbeta^(-1)*muBeta+xx'*y);
aStar = priorSig(1) + n/2;
bStar = priorSig(2) + .5*(y'*y-muStar'*Vstar^(-1)*muStar);

for iter = 1:iterations
    sig2 = gamrnd(aStar,1/bStar);
    
    bb = mvnrnd(muStar,Vstar);
    alpha = bb(1);
    beta = bb(2:end)';
    theta(iter).alpha = alpha;
    theta(iter).beta = beta;
    theta(iter).sig2 = sig2;
    % log(p(x,y|alpha,beta,sigma))
    llVec(iter) = llVec(iter) - .5*log(2*pi*sig2) - 1/(2*sig2)*sum((y-xx*[alpha;beta]).^2);
    % log(p(alpha,beta|sigma))
    llVec(iter) = llVec(iter) - sum((1./(2*sig2*priorBeta(:,2))).*(([alpha;beta]-priorBeta(:,1)).^2));
    % log(p(sigma))
    llVec(iter) = llVec(iter) - (priorSig(1)+1)*log(1/sig2)-priorSig(2)/sig2 + (priorSig(1)*log(priorSig(2))) - gammaln(priorSig(1)); 
    yHat = xx*[alpha;beta];
    theta(iter).yHat = yHat;
    %[alpha;beta;sig2]'
end

% figure
% plot(llVec)
% 
% figure
% plot(y,yHat,'g.')
% 
% figure
% hold on
% for iter = 1:iterations
%     ss = theta(iter).sig2;
%     plot(iter,ss,'.')
% end
% 
% figure
% hold on
% for iter = 1:iterations
%     yHat = theta(iter).yHat;
%     yErr = mean((y-yHat).^2);
%     plot(iter,yErr,'r.')
% end
% 
% Vstar
% muStar
% aStar
% bStar
    