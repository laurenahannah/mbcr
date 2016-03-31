function [alpha, beta, K] = CAP_CV2(x,y,numFold)
% Use cross validation to choose K!
% For convex functions only!


[n,d] = size(x);

keepGoing = true;
errOld = inf;
errVec = [];
xSize = floor(n/numFold);
tStart = tic;
k = 1;
while (keepGoing)
    ev = zeros(numFold,1);
    for i = 1:numFold
        xInd = (i-1)*xSize+1:i*xSize;
        xxTest = x(xInd,:);
        yyTest = y(xInd);
        indTrain = setdiff([1:n],xInd);
        xxTrain = x(indTrain,:);
        yyTrain = y(indTrain);
        %[size(xxTrain), size(xxTest)]
        [a, b, g, KK] = CAP_div(xxTrain,yyTrain,k);
        yHat = convexEval(a,b,xxTest);
        ev(i) = mean((yyTest - yHat).^2);
    end
    errVec = [errVec; mean(ev)];
    if (k == 1)
        errOld = errVec(k);
    %elseif (errVec(k) >= errOld)
    %    keepGoing = false;
    elseif ((k > n/(k*.8*(d+1)))||(k > 20))
        keepGoing = false;
    else
        errOld = errVec(k);
    end
    k = k + 1;
end

errVec'

[val, K] = min(errVec);
[alpha, beta, g, KK] = CAP_div(x,y,K);
tEnd = toc(tStart);

disp('Time elapsed:')
disp(tEnd)

disp('K')
disp([K])

        



function [alpha beta g K] = CAP_div(x,y,divisions)

% This makes a convex regression tree for input data x,y using least
% squares regression in each partition
% Inputs:
%  - x (covariate training data)
%  - y (response training data)
%  - divisions (maximum number of partitions... it may terminate before
%  this, but hey, you don't want it cooking forever)
%  - lambda (tolerance parameter)
%  - convexFlag (specifices whether function is convex or convace: ittakes 
%  values 0 for convex functions and 1 for concave functions)
% Outputs:
%  - alpha (row vector of intercepts)
%  - beta (matrix where columns are slopes of a partition)
%  - g (predicted output for the training data)
%  - K (number of components in the output)

convexFlag = 0;


[n, d] = size(x);
knotNum = 20; % Number of knots in each direction
minNode = 1.2*(d+1); % The minimum number of observations for a regression to be fit

if isempty(divisions)
    divisions = 100;
end


margTime = zeros(divisions,1);
margSSE = zeros(divisions,1);
margErr = zeros(divisions,1);

idList = ones(n,1); % Everything starts in the same partition

partCounter = 0;
%feasible = false;

keepGoing = true;

if minNode >= n
    ab = zeros(d+1,1);
    ab(1) = mean(y);
else
    ab = regress(y,[ones(n,1), x]);
end
alpha = ab(1);
beta = ab(2:end);
g = [ones(n,1),x]*ab;

kOld = length(alpha);
while (keepGoing)
    numPart = max(idList);
    sse = (sum((g-y).^2)+10)*ones(numPart*d*(knotNum+1)+1,1);
    sse(end) = sum((g-y).^2);
    
    sseLast = sum((g-y).^2);
    
    tstart = tic;
    dataStruct = [];
    for i = 1:numPart*d*(knotNum+1)+1
        dataStruct(i).beta = [];
        dataStruct(i).alpha = [];
        dataStruct(i).g = [];
        dataStruct(i).idList = [];
    end
    dataStruct(end).beta = beta;
    dataStruct(end).alpha = alpha;
    dataStruct(end).g = g;
    dataStruct(end).idList = idList;
    counter = 1;
    for i = 1:numPart
        inds = find(idList == i);
        xHat = x(inds,:);
        yHat = y(inds);
        ni = length(inds);
        betaH = beta;
        alphaH = alpha;
        betaH(:,i) = [];
        alphaH(:,i) = [];
        if (ni >= 2*minNode)
            for j = 1:d
                % Create grid for searching
                minVec = min(xHat(:,j));
                maxVec = max(xHat(:,j));
                if (maxVec - minVec < 0.0001) % We need to jitter
                    xHat(:,j) = xHat(:,j) + 0.0001*randn(ni,1);
                    minVec = min(xHat(:,j));
                    maxVec = max(xHat(:,j));
                end
                knotStep = (maxVec - minVec)/knotNum;
                knotMat = zeros(1,knotNum+1);
                knotMat = minVec:knotStep:maxVec;
                for k = 2:knotNum
                    % Make sure that both parts of the partition have
                    % enough data
                    xHat1Ind = find(xHat(:,j) <= knotMat(k));
                    xHat2Ind = find(xHat(:,j) > knotMat(k));
                    n1 = length(xHat1Ind);
                    n2 = length(xHat2Ind);
                    if ((n1 > minNode)&&(n2 > minNode))
                        % Then we can compute betas
                        ab1 = regress(yHat(xHat1Ind),[ones(n1,1),xHat(xHat1Ind,:)]);
                        ab2 = regress(yHat(xHat2Ind),[ones(n2,1), xHat(xHat2Ind,:)]);
                        betaHat = [betaH, ab1(2:end), ab2(2:end)];
                        alphaHat = [alphaH, ab1(1), ab2(1)];
                        if convexFlag == 0
                            [gg, iList] = max([ones(n,1),x]*[alphaHat; betaHat],[],2);
                        else
                            [gg, iList] = min([ones(n,1),x]*[alphaHat; betaHat],[],2);
                        end
                        dataStruct(counter).beta = betaHat;
                        dataStruct(counter).alpha = alphaHat;
                        dataStruct(counter).g = gg;
                        dataStruct(counter).idList = iList;
                        sse(counter) = sum((y-gg).^2);
                    else
                        sse(counter) = sse(end) + 10;
                    end
                    counter = counter + 1;
                end
            end
        end
    end
    % Now to choose the best
    [aa, bb] = min(sse);
    beta = dataStruct(bb).beta;
    alpha = dataStruct(bb).alpha;
    g = dataStruct(bb).g;
    idList = dataStruct(bb).idList;
    
    % Update other linear models
    totNum = max(idList);
    betaHat = [];
    alphaHat = [];
    for i = 1:totNum
        iHat = find(idList==i);
        ni = length(iHat);
        if (ni >= minNode)
            ab = regress(y(iHat),[ones(ni,1),x(iHat,:)]);
            alphaHat = [alphaHat, ab(1)];
            betaHat = [betaHat, ab(2:end)];
        else
            alphaHat = [alphaHat, alpha(i)];
            betaHat = [betaHat, beta(:,i)];
        end
    end
    beta = betaHat;
    alpha = alphaHat;
    if (convexFlag == 0)
        [g, idList] = max([ones(n,1),x]*[alpha; beta],[],2);
    else
        [g, idList] = min([ones(n,1),x]*[alpha; beta],[],2);
    end
    
    % Record stuff
    tend = toc(tstart);
    
    margTime(partCounter+1) = tend;
    
    SSE(partCounter+1) = sum((y-g).^2);
    
    if partCounter == 0
        theDiff = SSE(1);
    else
        theDiff = SSE(partCounter) - SSE(partCounter+1);
    end
    
    %disp(theDiff)
    
    partCounter = partCounter + 1;
    if ((length(alpha)>=divisions))
        keepGoing = false;
    end
    if (length(alpha)<= kOld)
        % Don't get stuck in an infinite loop... infinite loop... infinite
        % loop...
        keepGoing = false;
    end
    kOld = length(alpha);
end
                        
        
K = length(alpha);

function y = convexEval(alpha,beta,x)

% This function takes the alphas and betas exported by convexTree, fits a
% function and then evaluates it at x, returning y
[n,d] = size(x);

[y, iList] = max([ones(n,1),x]*[alpha; beta],[],2);