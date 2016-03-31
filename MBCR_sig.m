function abStruct = MBCR_sig(x,y,burnin,nTot)
% MBCR implements Multivariate Bayesian Convex Regression and outputs a
% structure array of slopes and intercepts for each realization.
%
% We have no BIC penalty.
%
% We assume a local sigma.
% 
% Inputs:
%   - x                 : x training data
%   - y                 : y training data
%   - burnin            : number of burnin iterations
%   - nTot              : total number of iterations
% Outputs:
%   - abStruct          : a structure array of [alpha, beta] for each
%   realization

pulse = 1;



outTime = [burnin:pulse:nTot];
nRealize = length(outTime);
yTime = [];

if (nargin == 8)
    yOut = [];
end

[n,d] = size(x);
varBeta = 5;
varAlpha = 5;
priorBeta = [zeros(d+1,1), varBeta*ones(d+1,1)];
priorBeta(1,2) = varAlpha;

varBetaP = 5;
varAlphaP = 5;
priorBetaP = [zeros(d+1,1), varBetaP*ones(d+1,1)];
priorBetaP(1,2) = varAlphaP;

priorSig = [1, 1];

propVarA = .1;
propVarB = .1*ones(d,1);

llVec = zeros(nTot,1);

% Prior on the number of clusters, geometric with parameter pGeo
pLambda = 20;
c = 0.4; % For birth/death probabilities


% Get an initial partitioning
%[xStruct, alpha, beta, K] = basisCAP(x,y,100,n*3);
xStruct = [];

alpha = zeros(1,1);
beta = zeros(d,1);
[thetaStar, lll] = bayesianLinearRegression(x,y,1);
alpha = thetaStar(1).alpha;
beta = thetaStar(1).beta;
K = 1;

tStart = tic;
% Get a variance
sig2 = drawSigma(alpha,beta,x,y,priorSig);

sigVec = sig2;

K

% Use initial partitioning to get proposal regions

Vbeta = diag(priorBetaP(:,2));
muBeta = priorBeta(:,1);

if (d > 4)
    permNumber = 4; % Number of permutations
else permNumber = d;
end
partNum = 5;

partVec = drawDirichlet(10*ones(1,partNum));


if (d > 1)
    permMat = zeros(permNumber,d);
    permMat = randn(permNumber,d);
else
    permMat = 1;
end

divisions = length(partVec);
dims = length(permMat(:,1));

% Make structure for current alpha, beta
[currentX, addX, removeX, addProb, removeProb] = makeProposal(alpha,beta,permMat,partVec,muBeta,Vbeta,x,y,priorSig,muBeta,priorBeta(:,2),pLambda);

choiceVec = zeros(K,1);

disp('Line 92')
addProb
removeProb
size(currentX)
size(addX)


nAdd = 0;
nRemove = 0;

abCounter = 1;
numAcc = 0;

changeVec = [];
selectVec = [];

% Now to do reversivble jump MCMC
t = 1;
tempStruct = [];
tLag = 0;
tPick = 0;
timeMax = 100; % Acceptance rate < 1/3%
tempStruct(1).alpha = alpha;
tempStruct(1).beta = beta;
tempStruct(1).currentX = currentX;
tempStruct(1).addX = addX;
tempStruct(1).removeX = removeX;
tempStruct(1).addProb = addProb;
tempStruct(1).removeProb = removeProb;
tempStruct(1).sigVec = sigVec;

while (numAcc < nTot) % Do nTot iterations
    K  = length(alpha);
    
    pAdd = c*min(1,pLambda/(K));
    pRemove = c*min(1,(K-1)/pLambda);
    pChange = 1 - pAdd - pRemove;
    if (pChange < 0)
        disp('Change c!')
        disp([pAdd, pRemove,pChange])
    %else
    %    disp([pAdd, pRemove, pChange])
    end
    
    partVec = drawDirichlet(ones(1,partNum));
    permMat = randn(permNumber,d);
    
%     if length(addX) ~= length(addProb)
%         length(alpha)
%         length(addX)
%         length(addProb)
%         addProb
%         t
%     end
    [alphaProp, betaProp, sigProp] = drawObservation(x,y,currentX, addX, removeX, addProb, removeProb,pAdd,pRemove,pChange,priorBeta(:,1),priorBeta(:,2),muBeta,Vbeta,pLambda,priorSig,permMat,partVec);
    
    changeComp = false;
    addComp = false;
    removeComp = false;
    
    KK = length(alphaProp);
    if (KK == K)
        changeComp = true;
    elseif (KK > K)
        addComp = true;
    else
        removeComp = true;
    end
    
    [gOld, idListOld] = max([ones(n,1),x]*[alpha;beta],[],2);
    [gNew, idListNew] = max([ones(n,1),x]*[alphaProp;betaProp],[],2);
    
    [currentXP, addXP, removeXP, addProbP, removeProbP] = makeProposal(alphaProp,betaProp,permMat,partVec,muBeta,Vbeta,x,y,priorSig,muBeta,priorBeta(:,2),pLambda);
    
    [nSig, dSig] = size(sigProp);
    if (dSig > 1)
        sigProp = sigProp';
    end
    
    [nSig2, dSig2] = size(sigVec);
    if (dSig2 > 1)
        sigVec = sigVec';
    end
    
    if (changeComp)
        pp = prod(sqrt(sigVec(idListOld)./sigProp(idListNew)))*exp(-1/(2)*sum(((y-gNew).^2)./sigProp(idListNew))+1/(2)*sum(((y-gOld).^2)./sigVec(idListOld)));
        logPiTop = 0;
        logPiBottom = 0;
        for i = 1:K
            logPiTop = logPiTop - sum((1./(2*priorBeta(:,2))).*(([alphaProp(i); betaProp(:,i)]-priorBeta(:,1)).^2));
            logPiTop = logPiTop - (priorSig(1)+1)*log(1/sigProp(i))-priorSig(2)/sigProp(i) + (priorSig(1)*log(priorSig(2))) - gammaln(priorSig(1));
            logPiBottom = logPiBottom - sum((1./(2*priorBeta(:,2))).*(([alpha(i);beta(:,i)]-priorBeta(:,1)).^2));
            logPiBottom = logPiBottom - (priorSig(1)+1)*log(1/sigVec(i))-priorSig(2)/sigVec(i) + (priorSig(1)*log(priorSig(2))) - gammaln(priorSig(1));
        end
        probPi = exp(logPiTop - logPiBottom);
    elseif (addComp)
        pp = prod(sqrt(sigVec(idListOld)./sigProp(idListNew)))*exp(-1/(2)*sum(((y-gNew).^2)./sigProp(idListNew))+1/(2)*sum(((y-gOld).^2)./sigVec(idListOld)));
        logPiTop = 0;
        logPiBottom = 0;
        for i = 1:K
            logPiTop = logPiTop - sum((1./(2*priorBeta(:,2))).*(([alphaProp(i); betaProp(:,i)]-priorBeta(:,1)).^2));
            logPiTop = logPiTop - (priorSig(1)+1)*log(1/sigProp(i))-priorSig(2)/sigProp(i) + (priorSig(1)*log(priorSig(2))) - gammaln(priorSig(1));
            logPiBottom = logPiBottom - sum((1./(2*priorBeta(:,2))).*(([alpha(i);beta(:,i)]-priorBeta(:,1)).^2));
            logPiBottom = logPiBottom - (priorSig(1)+1)*log(1/sigVec(i))-priorSig(2)/sigVec(i) + (priorSig(1)*log(priorSig(2))) - gammaln(priorSig(1));
        end
        logPiTop = logPiTop - sum((1./(2*priorBeta(:,2))).*(([alphaProp(KK); betaProp(:,KK)]-priorBeta(:,1)).^2));
        logPiTop = logPiTop - (priorSig(1)+1)*log(1/sigProp(KK))-priorSig(2)/sigProp(KK) + (priorSig(1)*log(priorSig(2))) - gammaln(priorSig(1));
        probPi = exp(logPiTop - logPiBottom);
    else
        pp = prod(sqrt(sigVec(idListOld)./sigProp(idListNew)))*exp(-1/(2)*sum(((y-gNew).^2)./sigProp(idListNew))+1/(2)*sum(((y-gOld).^2)./sigVec(idListOld)));
        logPiTop = 0;
        logPiBottom = 0;
        for i = 1:KK
            logPiTop = logPiTop - sum((1./(2*priorBeta(:,2))).*(([alphaProp(i); betaProp(:,i)]-priorBeta(:,1)).^2));
            logPiTop = logPiTop - (priorSig(1)+1)*log(1/sigProp(i))-priorSig(2)/sigProp(i) + (priorSig(1)*log(priorSig(2))) - gammaln(priorSig(1));
            logPiBottom = logPiBottom - sum((1./(2*priorBeta(:,2))).*(([alpha(i);beta(:,i)]-priorBeta(:,1)).^2));
            logPiBottom = logPiBottom - (priorSig(1)-1)*log(1/sigVec(i))-priorSig(2)/sigVec(i) + (priorSig(1)*log(priorSig(2))) - gammaln(priorSig(1));
        end
        logPiBottom = logPiBottom - sum((1./(2*priorBeta(:,2))).*(([alpha(K); beta(:,K)]-priorBeta(:,1)).^2));
        logPiBottom = logPiBottom - (priorSig(1)+1)*log(1/sigVec(K))-priorSig(2)/sigVec(K) + (priorSig(1)*log(priorSig(2))) - gammaln(priorSig(1));
        probPi = exp(logPiTop - logPiBottom);
    end
    
    pAddP = c*min(1,pLambda/(KK));
    pRemoveP = c*min(1,(KK-1)/pLambda);
    pChangeP = 1 - pAddP - pRemoveP;
    
    [qFacT, qT] = getWeight(alpha,beta,currentXP,addXP,removeXP,addProbP,removeProbP,pAddP,pRemoveP,pChangeP,sigVec);
    [qFacB, qB] = getWeight(alphaProp,betaProp,currentX,addX,removeX,addProb,removeProb,pAdd,pRemove,pChange,sigProp);
        
    
    % Accept or reject
    aProb = pp*probPi*qT/qB*exp(qFacT-qFacB);
    u = rand;
    accepted = false;
    if (u <= aProb) % We accept
        accepted = true;
        alpha = alphaProp;
        sigVec = sigProp;
        if (KK > K)
            nAdd = nAdd+1;
        elseif (KK < K)
            nRemove = nRemove+1;
        end
        beta = betaProp;
        K = length(alpha);
        tLag = max(tLag-1,0);
        if (tLag == 0)
            numAcc = numAcc+1;
        end
        llVec(numAcc) = logPiTop-1/(2)*sum(((y-gNew).^2)./sigVec(idListNew))-(n/2)*log(K*(d+1));
        
        tempStruct(numAcc).alpha = alpha;
        tempStruct(numAcc).beta = beta;
        tempStruct(numAcc).currentX = currentX;
        tempStruct(numAcc).addX = addX;
        tempStruct(numAcc).removeX = removeX;
        tempStruct(numAcc).addProb = addProb;
        tempStruct(numAcc).removeProb = removeProb;
        tempStruct(numAcc).sigVec = sigVec;
    else
        tPick = tPick + 1;
    end
    
    
    if (tPick >= timeMax) % Restart the chain
        tLag = 3; % Go 3 steps before we count it restarted
        tPick = 0;
        if (numAcc > 30) % Randomly select a state between 30 and current
            iii = randint(1,1,[30,numAcc]);
        else
            iii = randint(1,1,[1,numAcc]);
        end
        alpha = tempStruct(iii).alpha;
        beta = tempStruct(iii).beta;
        currentX = tempStruct(iii).currentX;
        addX = tempStruct(iii).addX;
        removeX = tempStruct(iii).removeX;
        addProb = tempStruct(iii).addProb;
        removeProb = tempStruct(iii).removeProb;
        sigVec = tempStruct(iii).sigVec;
    end
    
    if (mod(t,1)==0)
        [currentX, addX, removeX, addProb, removeProb] = makeProposal(alpha,beta,permMat,partVec,muBeta,Vbeta,x,y,priorSig,muBeta,priorBeta(:,2),pLambda);
    end        
    
    if (((numAcc >= burnin)&&(mod(numAcc,pulse)==0))&&((accepted)&&(tLag==0)))
        % Record for output
        abStruct(abCounter).alpha = alpha;
        abStruct(abCounter).beta = beta;
        abStruct(abCounter).K = K;
        abStruct(abCounter).acc = numAcc;
        abStruct(abCounter).ll = llVec;
        abCounter = abCounter+1;
        %disp(t)
        %disp([K, mean((y-g).^2), numAcc])
        if (nargin == 8)
            nTest = length(yTest);
            yHat = max([ones(nTest,1), xTest]*[alpha; beta],[],2);
            yOut = [yOut, yHat];
        end
        
    end
    
    
    K = length(alpha);
    
    if ((numAcc/t*100)<3)
        partNum = 15;
        %permNumber = 10;
    elseif ((numAcc/t*100)>10)
        partNum = 5;
        %permNumber = 2;
    else
        partNum = 10;
        %permNumber = 5;
    end
    
    if((mod(numAcc,25*pulse)==0)&&(accepted))
        aStar = currentX.aStar;
        bStar = currentX.bStar;
        t
        numAcc
        [K, numAcc/t*100]
        [pChange, pAdd, pRemove, changeComp, addComp, removeComp]
        %[aStar;bStar]
        [sigVec';alpha;beta]
        aProb
        [mean((y-gNew).^2), mean((y-gOld).^2)]
        if ((nargin == 6)&&(~isempty(yOut)))
            disp('Current MSE, MSE of Average:')
            res = yTest - yHat;
            yyHat = mean(yOut,2);
            res2 = yTest - yyHat;
            disp([mean(res.^2), mean(res2.^2)])
        end
    end
    t = t+1;
    
end

tEnd = toc(tStart);
disp('Time in Seconds:')
disp(tEnd)

nAdd
nRemove
numAcc

%     figure
%     hold on
%     fMBCR2 = @(x)fMBCR(x,abStruct);
%     for i = 1:n
%         plot(sum(x(i,:)),y(i),'r.')
%         plot(sum(x(i,:)),fMBCR2(x(i,:)),'.')
%     end

if (nargin == 8)
    disp('MSE of Average:')
    yyHat = mean(yOut,2);
    res2 = yTest - yyHat;
    disp(mean(res2.^2))
end
    
    

%==========================================================================
% Log likelihood for NIG prior with variance parameter integrated out
%==========================================================================
function logLike = llNIG(beta,mu,varB,priorSig)
% This assumes a diagonal covariance matrix
d = length(beta);
V = diag(varB);
Sigma = (priorSig(1)/priorSig(2))*V;
nu = 2*priorSig(1);

logLike = gammaln((nu+d)/2)-gammaln(nu/2)-(d)/2*log(pi)-.5*log(det(nu*Sigma));
logLike = logLike -((nu+d)/2)*log(1+1/nu*(beta-mu)'*(Sigma^(-1))*(beta-mu));

function logLike = llBeta(beta,mu,varB,lambda)
% Creates log likelihood function for pi(alpha,beta,K)
[d,k] = size(beta);
V = diag(varB);
logLike = -lambda + (k-1)*log(lambda)-sum(1:(k-1));
for i = 1:k
    bb = beta(:,i);
    logLike = logLike - (d/2)*log(2*pi) - log(det(V)) - .5*(bb-mu)'*(V^(-1))*(bb-mu);
end



%==========================================================================
% Make basis regions
%==========================================================================

function sig2 = drawSigma(alpha,beta,x,y,sigPrior)
[n,d] = size(x);

[g, iList] = max([ones(n,1),x]*[alpha; beta],[],2);
aStar = sigPrior(1) + n/2;
bStar = sigPrior(2) + .5*sum((y-g).^2);
sig2 = 1/(gamrnd(aStar,1/bStar));

%==========================================================================
function [currentX, addX, removeX, addProb, removeProb] = makeProposal(alpha,beta,permMat,partVec,muBeta,Vbeta,x,y,sigPrior,muBPrior,varBPrior,lambda)
[n,d] = size(x);

K = length(alpha);
[g, idList] = max([ones(n,1),x]*[alpha; beta],[],2);

partNum = length(partVec);
divisions = length(permMat(:,1));
if isempty(divisions)
    divisions = 1;
end
    

currentX = [];
removeX = [];
addX = [];
removeProb = zeros(K,1); % Proportional to number of observations assoicated
addProb = []; % Proportional to LS likelihood

llAddProb = [];

gVec = [];

currentX.id = idList;
currentX.g = g;
muVec = zeros(d+1,K); % Vector of means
VVec = zeros(d+1,K*(d+1)); % Due to Matlab stupidity, we storing the
% covariance matrices in a vector rather than a three dimensional array
VInvVec = zeros(d+1,K*(d+1)); % Store covariance inverses

aStar = zeros(1,K);
bStar = zeros(1,K);


for i = 1:K
    vec = find(idList == i);
    ni = length(vec);
    if ni > 0
        removeProb(i) = 1/ni;
    else
        removeProb(i) = 1/.25;
    end
    xx = [ones(ni,1), x(vec,:)];
    yy = y(vec);
    Vstar = (Vbeta^(-1) + xx'*xx)^(-1);
    muStar = Vstar*(Vbeta^(-1)*muBeta+xx'*yy);
    muVec(:,i) = muStar;
    j = (i-1)*(d+1);
    VVec(:,(j+1):(j+d+1)) = Vstar;
    vInv = Vstar^(-1);
    VInvVec(:,(j+1):(j+d+1)) = vInv;
    aStar(i) = sigPrior(1) + ni/2;
    bStar(i) = sigPrior(2) + .5*(yy'*yy-muStar'*vInv*muStar);
end

removeProb = removeProb/sum(removeProb);

currentX.V = VVec;
currentX.mu = muVec;
currentX.VInv = VInvVec;
currentX.aStar = aStar;
currentX.bStar = bStar;

% Make structure for removal of hyperplane

for i = 1:K
    if (K == 1) % Don't break computer if we remove the only hyperplane
        removeX(1).mu=[];
        removeX(1).V = [];
        removeX(1).VInv = [];
    else
        % We remove hyperplane i
        alphaP = alpha;
        betaP = beta;
        alphaP(i) = [];
        betaP(:,i) = [];
        [g2, idList2] = max([ones(n,1),x]*[alphaP; betaP],[],2);
        removeX(i).id = idList2;
        removeX(i).g = g2;
        muVec = zeros(d+1,(K-1)); % Vector of means
        VVec = zeros(d+1,(K-1)*(d+1)); % Due to Matlab stupidity, we storing the
        % covariance matrices in a vector rather than a three dimensional array
        VInvVec = zeros(d+1,(K-1)*(d+1)); % Store covariance inverses
        aStar = zeros(1,K-1);
        bStar = zeros(1,K-1);

        jCounter = 1;
        for j = 1:K-1
                vec = find(idList2 == j);
                ni = length(vec);
                xx = [ones(ni,1), x(vec,:)];
                yy = y(vec);
                Vstar = (Vbeta^(-1) + xx'*xx)^(-1);
                vInv = Vstar^(-1);
                muStar = Vstar*(Vbeta^(-1)*muBeta+xx'*yy);
                muVec(:,jCounter) = muStar;
                aStar(jCounter) = sigPrior(1) + ni/2;
                bStar(jCounter) = sigPrior(2) + .5*(yy'*yy-muStar'*vInv*muStar);
                k = (jCounter-1)*(d+1);
                VVec(:,(k+1):(k+d+1)) = Vstar;
                VInvVec(:,(k+1):(k+d+1)) = vInv;
                jCounter = jCounter + 1;
        end

        removeX(i).V = VVec;
        removeX(i).mu = muVec;
        removeX(i).VInv = VInvVec;
        removeX(i).aStar = aStar;
        removeX(i).bStar = bStar;
    end
end

% Make a structure for the addition of hyperplanes
addCounter = 1;
llAddProb = 0;

for i = 1:K % Loop thru each hyperplane
    inds = find(idList == i);
    idListHat = idList;
    xHat = x(inds,:);
    yHat = y(inds);
    ni = length(inds);
    betaH = beta;
    alphaH = alpha;
    betaH(:,i) = [];
    alphaH(:,i) = [];
    
    muVecH = currentX.mu;
    VVecH = currentX.V;
    VInvVecH = currentX.VInv;
    aStar = currentX.aStar;
    bStar = currentX.bStar;
        
    muVecH(:,i) = [];
    VVecH(:,((i-1)*(d+1)+1):(i*(d+1))) = [];
    VInvVecH(:,((i-1)*(d+1)+1):(i*(d+1))) = [];
    aStar(i) = [];
    bStar(i) = [];
    for j = 1:divisions % Loop thru each dimension
        % Create grid for searching
        xxHat = xHat*permMat(j,:)'; % Permuted data
        minVec = min(xxHat);
        maxVec = max(xxHat);
        if (maxVec - minVec < 0.01) % We need to jitter
            xxHat = xxHat + 0.01*randn(ni,1);
            minVec = min(xxHat);
            maxVec = max(xxHat);
        end
        if isempty(minVec)
            minVec = -1;
        end
        if isempty(maxVec)
            maxVec = 1;
        end
        knotMatHat = (maxVec-minVec)*partVec;
        knotMat = minVec + cumsum(knotMatHat);
%         if length(knotMat) < 1
%             minVec
%             maxVec
%             knotStep
%             length(inds)
%         end
        for k = 1:partNum-1
            % Make sure that both parts of the partition have
            % enough data
            alphaP = alphaH;
            betaP = betaH;
            idListHat = idList;
            xHat1Ind = find(xxHat <= knotMat(k));
            xHat2Ind = find(xxHat >= knotMat(k));
            % Make mu and V
            if isempty(xHat1Ind)
                niLo = 0;
                xx1 = [];
                yy1 = [];
                Vstar1 = Vbeta;
                muStar1 = Vstar1*(Vbeta^(-1)*muBeta);
                aStar1 = sigPrior(1) + niLo/2;
                bStar1 = sigPrior(2);
            else
                niLo = length(xHat1Ind);
                xx1 = [ones(niLo,1), xHat(xHat1Ind,:)];
                yy1 = yHat(xHat1Ind);
                Vstar1 = (Vbeta^(-1) + xx1'*xx1)^(-1);
                muStar1 = Vstar1*(Vbeta^(-1)*muBeta+xx1'*yy1);
                aStar1 = sigPrior(1) + niLo/2;
                bStar1 = sigPrior(2) + .5*(yy1'*yy1-muStar1'*Vstar1^(-1)*muStar1);
            end
            if isempty(xHat2Ind)
                niHi = 0;
                xx2 = [];
                yy2 = [];
                Vstar2 = Vbeta;
                muStar2 = Vstar2*(Vbeta^(-1)*muBeta);
                aStar2 = sigPrior(1) + niHi/2;
                bStar2 = sigPrior(2);
            else
                niHi = length(xHat2Ind);
                xx2 = [ones(niHi,1), xHat(xHat2Ind,:)];
                yy2 = yHat(xHat2Ind);
                Vstar2 = (Vbeta^(-1) + xx2'*xx2)^(-1);
                muStar2 = Vstar2*(Vbeta^(-1)*muBeta+xx2'*yy2);
                aStar2 = sigPrior(1) + niHi/2;
                bStar2 = sigPrior(2) + .5*(yy2'*yy2-muStar2'*Vstar2^(-1)*muStar2);
            end
            
            muVecP = [muVecH, muStar1, muStar2];
            VVecP = [VVecH, Vstar1, Vstar2];
            VInvVecP = [VInvVecH, Vstar1^(-1), Vstar2^(-1)];
            aStar = [aStar, aStar1, aStar2];
            bStar = [bStar, bStar1, bStar2];
            
            addX(addCounter).mu = muVecP;
            addX(addCounter).V = VVecP;
            addX(addCounter).VInv = VInvVecP;
            addX(addCounter).aStar = aStar;
            addX(addCounter).bStar = bStar;
            
            % Get log likelihood
            if (niLo < 1.2*(d+1))
                alphaP = [alphaP, muBeta(1)];
                betaP = [betaP, muBeta(2:end)];
            else
                ab1 = ridge(yHat(xHat1Ind),[ones(niLo,1),xHat(xHat1Ind,:)],1);
                alphaP = [alphaP, ab1(1)];
                betaP = [betaP, ab1(2:end)];
            end
            if (niHi < 1.2*(d+1))
                alphaP = [alphaP, muBeta(1)];
                betaP = [betaP, muBeta(2:end)];
            else
                ab1 = ridge(yHat(xHat2Ind),[ones(niHi,1),xHat(xHat2Ind,:)],1);
                alphaP = [alphaP, ab1(1)];
                betaP = [betaP, ab1(2:end)];
            end
            [gg, iidList] = max([ones(n,1),x]*[alphaP; betaP],[],2);
            %llAddProb(addCounter) = -1/(2*sig2)*sum((y-gg).^2);
            %llAddProb(addCounter) = llAddProb(addCounter) + llBeta([alphaP;betaP],muBPrior,varBPrior,lambda);
            %if (ni > 0)
            %    llAddProb(addCounter) = llAddProb(addCounter) + log(ni);
            %else
            %    llAddProb(addCounter) = llAddProb(addCounter) + log(.25);
            %end
            KK = max(iidList);
            llAddProb = [llAddProb; 0];
            for kCounter = 1:KK
                nnVec = find(iidList == kCounter);
                nn = length(nnVec);
                if (nn < 1)
                    nn = .25;
                end
                llAddProb(addCounter) = llAddProb(addCounter)+log(nn);
            end
                
%             if (niLo == 0)
%                 niLoNew = .25;
%             else
%                 niLoNew = niLo;
%             end
%             if (niHi == 0)
%                 niHiNew = .25;
%             else
%                 niHiNew = niHi;
%             end
%             llAddProb(addCounter) = log(niLoNew) + log(niHiNew);
            gVec(addCounter) = sum((y-gg).^2);
            addCounter = addCounter + 1;
        end
    end
end

%size(currentX)
%size(addX)


% Normalize probabilities
llMax = max(llAddProb);
llAddProb = llAddProb - llMax;
addProb = exp(llAddProb)'/sum(exp(llAddProb));
addProb = addProb(1:end-1);
            
%==========================================================================
% Draw observation from proposal dist and provide log weight

function [alpha, beta, sigVec] = drawObservation(x,y,currentX, addX, removeX, addProb, removeProb,pAdd,pRemove,pChange,muBPrior,varBPrior,muBeta,Vbeta,lambda,sigPrior,permMat,partVec)

[n, d] = size(x);

alpha = [];
beta = [];
sigVec = [];

% Boolean variables to hold decision
addComp = false;
removeComp = false;
changeComp = false;

u = rand;

if (u <= pAdd)
    addComp = true;
elseif (u <= pAdd + pRemove)
    removeComp = true;
else
    changeComp = true;
end

if (changeComp) % Draw from currentX
    mu = currentX.mu;
    V = currentX.V;
    aStar = currentX.aStar;
    bStar = currentX.bStar;
    kk = length(mu(1,:));
    for i = 1:kk
        m = mu(:,i);
        sig2 = 1/(gamrnd(aStar(i),1/bStar(i)));
        vv = V(:,((i-1)*(d+1)+1):(i*(d+1)))*sig2;
        bb = mvnrnd(m',vv);
        alpha = [alpha, bb(1)];
        beta = [beta, bb(2:end)'];
        sigVec = [sigVec, sig2];
    end
elseif (removeComp) % Draw from removeX
    cChoice = multinomialRnd(1,removeProb');
    if (isnan(cChoice(1)))
        disp('Line 681')
    end
    [aa, compChoice] = max(cChoice);
    mu = removeX(compChoice).mu;
    V = removeX(compChoice).V;
    aStar = removeX(compChoice).aStar;
    bStar = removeX(compChoice).bStar;
    kk = length(mu(1,:));
    for i = 1:kk
        m = mu(:,i);
        sig2 = 1/(gamrnd(aStar(i),1/bStar(i)));
        vv = V(:,((i-1)*(d+1)+1):(i*(d+1)))*sig2;
        bb = mvnrnd(m',vv);
        alpha = [alpha, bb(1)];
        beta = [beta, bb(2:end)'];
        sigVec = [sigVec, sig2];
    end
else % Draw from addX
    cChoice = multinomialRnd(1,addProb');
    if (isnan(cChoice(1)))
        disp('Line 701')
    end
    [aa, compChoice] = max(cChoice);
    %size(addX)
    mu = addX(compChoice).mu;
    V = addX(compChoice).V;
    aStar = addX(compChoice).aStar;
    bStar = addX(compChoice).bStar;
    kk = length(mu(1,:));
    for i = 1:kk
        m = mu(:,i);
        sig2 = 1/(gamrnd(aStar(i),1/bStar(i)));
        vv = V(:,((i-1)*(d+1)+1):(i*(d+1)))*sig2;
        bb = mvnrnd(m',vv);
        alpha = [alpha, bb(1)];
        beta = [beta, bb(2:end)'];
        sigVec = [sigVec, sig2];
    end
end

if(size(alpha)~=size(sigVec))
    disp('Problem with new sigma vector')
    disp(size(alpha))
    disp(size(sigVec))
    disp([changeComp, addComp, removeComp])
end

[g, idList] = max([ones(n,1),x]*[alpha; beta],[],2);

[currentXP, addXP, removeXP, addProbP, removeProbP] = makeProposal(alpha,beta,permMat,partVec,muBeta,Vbeta,x,y,sigPrior,muBPrior,varBPrior,lambda);

KK = length(alpha);
c = 0.4;
pAdd = c*min(1,lambda/(KK));
pRemove = c*min(1,(KK-1)/lambda);
pChange = 1 - pAdd - pRemove;
    

%==========================================================================
% Draw observation from proposal dist and provide log weight

function [qFactor, q] = getWeight(alpha,beta,currentX,addX,removeX,addProb,removeProb,pAdd,pRemove,pChange,sigVec)

[d, kNew] = size(beta);

mm = currentX.mu;

[dd, kOld] = size(mm);


% Boolean variables to hold decision
addComp = false;
removeComp = false;
changeComp = false;

if (kNew == kOld)
    changeComp = true;
elseif (kNew > kOld)
    addComp = true;
else
    removeComp = true;
end

qVec = [];

if (changeComp)
    mu = currentX.mu;
    V = currentX.V;
    aStar = currentX.aStar;
    bStar = currentX.bStar;
    VInv = currentX.VInv;
    qVec = zeros(kNew,1);
    for i = 1:kNew
        m = mu(:,i);
        vv = V(:,((i-1)*(d+1)+1):(i*(d+1)))*sigVec(i);
        vvInv = VInv(:,((i-1)*(d+1)+1):(i*(d+1)))*1/sigVec(i);
        normVal = [alpha(i); beta(:,i)]' - m';
        qVec(i) = -(d+1)/2*log(2*pi)-.5*log(det(vv))+(-1/sigVec(i)*(bStar(i)+.5*(normVal)*(vvInv)*(normVal)'))+aStar(i)*log(bStar(i))-gammaln(aStar(i))-(aStar(i)+(d+1)/2+1)*log(sigVec(i));
    end
    qVec = qVec + log(pChange);
    qFactor = - max(qVec);
    qqVec = qVec + qFactor;
    q = sum(exp(qqVec));
    
elseif (removeComp)
    qVec = zeros(length(removeProb),kNew);
    for j = 1:length(removeProb)
        mu = removeX(j).mu;
        V = removeX(j).V;
        aStar = removeX(j).aStar;
        bStar = removeX(j).bStar;
        VInv = removeX(j).VInv;
        for i = 1:kNew
            m = mu(:,i);
            vv = V(:,((i-1)*(d+1)+1):(i*(d+1)))*sigVec(i);
            vvInv = VInv(:,((i-1)*(d+1)+1):(i*(d+1)))/sigVec(i);
            normVal = [alpha(i); beta(:,i)]' - m';
            qVec(j,i) = -(d+1)/2*log(2*pi)-.5*log(det(vv))+(-1/sigVec(i)*(bStar(i)+.5*(normVal)*(vvInv)*(normVal)'))+aStar(i)*log(bStar(i))-gammaln(aStar(i))-(aStar(i)+(d+1)/2+1)*log(sigVec(i));
        end
        qVec(j,:) = qVec(j,:) + log(removeProb(j)) + log(pRemove);
    end
    qFactor = - max(max(qVec));
    qqVec = qVec + qFactor;
    q = sum(sum(exp(qqVec)));
else
    qVec = zeros(length(addX),kNew);
    for j = 1:length(addX)
        mu = addX(j).mu;
        V = addX(j).V;
        aStar = addX(j).aStar;
        bStar = addX(j).bStar;
        VInv = addX(j).VInv;
        for i = 1:kNew
            m = mu(:,i);
            vv = V(:,((i-1)*(d+1)+1):(i*(d+1)))*sigVec(i);
            vvInv = VInv(:,((i-1)*(d+1)+1):(i*(d+1)))/sigVec(i);
            normVal = [alpha(i); beta(:,i)]' - m';
            qVec(j,i) = -(d+1)/2*log(2*pi)-.5*log(det(vv))+(-1/sigVec(i)*(bStar(i)+.5*(normVal)*(vvInv)*(normVal)'))+aStar(i)*log(bStar(i))-gammaln(aStar(i))-(aStar(i)+(d+1)/2+1)*log(sigVec(i));
        end
        qVec(j,:) = qVec(j,:) + log(addProb(j)) + log(pAdd);
    end
    qFactor = - max(max(qVec));
    qqVec = qVec + qFactor;
    q = sum(sum(exp(qqVec)));
end


