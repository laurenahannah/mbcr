% Minimize a random convex function:
% y = x_{1:2}'*Q*x_{1:2} + epsilon,
% epsilon ~ N(0,sigma^2)
% Q = [1 , .2; .2, 1]    (Positive Definite)
% subject to: -1 <= x_i <= 1, i = 1,2

% WARNING:

%=============================
% THIS REQUIRES CVX TO RUN!!!!
%=============================

% cvx can be downloaded from Stephen Boyd's website

clear all

Q = [1,.2;.2,1];

% Offline:

xList = [];
xVal = [];


for ell = 1:50

    ell
nTrain = 100;
xx = 2*rand([nTrain,2]) - 1;
y = zeros(nTrain,1);
for i = 1:nTrain
    y(i) = xx(i,:)*Q*xx(i,:)' + sqrt(.1)*randn;
end
[n, d] = size(xx);

% Do MBCR
abStructMin2 = MBCR_sig(xx,y,500,1000);
% Make a function from this
nSamp = length(abStructMin2);

% Do CAP
% Use leave-one-out cross validation
[alpha, beta, K] = CAP_CV2(xx,y,5);
fCAP = @(x) max([1, x]*[alpha; beta]);

% Do LSE
% Use cvx
if nTrain < 600
tstart = tic;
cvx_begin
    variables yhat(n) g1(n) g2(n)
    minimize(norm(y(1:n)-yhat))
    subject to
        yhat*ones(1,n) >= ones(n,1)*yhat' + (ones(n,1)*g1').*(xx(1:n,1)*ones(1,n)-ones(n,1)*xx(1:n,1)') + (ones(n,1)*g2').*(xx(1:n,2)*ones(1,n)-ones(n,1)*xx(1:n,2)');
cvx_end
tend = toc(tstart);
disp('Time LSE:')
disp(tend)
gg = [g1, g2];
aa = yhat - sum(gg.*xx(1:n,:),2);
a = aa';
b = gg';
else
a = 0;
b = zeros(d,1);
end
fLSE = @(x) max([1, x]*[a; b]);

ell

fLSE2 = @(x) max([1; x]'*[a; b]);
fCAP2 = @(x) max([1; x]'*[alpha; beta]);

% Minimize
x0 = zeros(1,2);
%[xLSE, yLSE] = fmincon(fLSE,x0,[],[],[],[],-ones(d,1),ones(d,1))
%[xCAP, yCAP] = fmincon(fCAP,x0,[],[],[],[],-ones(d,1),ones(d,1))
%fMBCR2 = @(x)fMBCR(x,abStructMin);
fMBCR22 = @(x)fMBCR(x,abStructMin2);
%[xMBCR, yMBCR] = fmincon(fMBCR2,x0,[],[],[],[],-ones(d,1),ones(d,1))

fTrue = @(x) x*Q*x';
fTrue2 = @(x) x'*Q*x;
%[xTrue, yTrue] = fmincon(fTrue,x0,[],[],[],[],-ones(d,1),ones(d,1))

cvx_begin
    variables xLSE(2)
    minimize(fLSE2(xLSE))
    subject to
        xLSE <= ones(2,1);
        xLSE >= -ones(2,1);
cvx_end

cvx_begin
    variables xCAP(2)
    minimize(fCAP2(xCAP))
    subject to
        xCAP <= ones(2,1);
        xCAP >= -ones(2,1);
cvx_end

% cvx_begin
%     variables xMBCR(2)
%     minimize(fMBCR2(xMBCR))
%     subject to
%         xMBCR <= ones(2,1);
%         xMBCR >= -ones(2,1);
% cvx_end

cvx_begin
    variables xMBCR2(2)
    minimize(fMBCR22(xMBCR2))
    subject to
        xMBCR2 <= ones(2,1);
        xMBCR2 >= -ones(2,1);
cvx_end

cvx_begin
    variables xTrue(2)
    minimize(fTrue2(xTrue))
    subject to
        xTrue <= ones(2,1);
        xTrue >= -ones(2,1);
cvx_end

xList = [xList; [xLSE', xCAP', xMBCR2', xTrue']]
xVal = [xVal; [fTrue2(xLSE), fTrue2(xCAP), fTrue2(xMBCR2)]]

xMesh = -1:.05:1;
sizeMesh = length(xMesh);
zCAP = zeros(sizeMesh);
zMBCR = zeros(sizeMesh);
zMBCRsig = zeros(sizeMesh);
zTrue = zeros(sizeMesh);
zLSE = zeros(sizeMesh);
for i = 1:sizeMesh
    for j = 1:sizeMesh
        zCAP(i,j) = fCAP([xMesh(i),xMesh(j)]);
        zMBCRsig(i,j) = fMBCR22([xMesh(i),xMesh(j)]);
        zTrue(i,j) = fTrue([xMesh(i),xMesh(j)]);
        zLSE(i,j) = fLSE([xMesh(i),xMesh(j)]);
    end
end
resCAP = zCAP - zTrue;
resMBCR = zMBCR - zTrue;
resLSE = zLSE - zTrue;
resStruct(ell).CAP = resCAP;
resStruct(ell).MBCR = resMBCR;
resStruct(ell).LSE = resLSE;
end

save xOptimize100_50_v2.dat xList -ascii;



figure
subplot(2,2,2)
title('LSE Function')
meshc(xMesh,xMesh,zLSE)
hold on
plot3(xx(:,1),xx(:,2),y,'k.')

subplot(2,2,4)
title('CAP Function')
meshc(xMesh,xMesh,zCAP)
hold on
plot3(xx(:,1),xx(:,2),y,'k.')

subplot(2,2,3)
title('MBCR Mean Function')
meshc(xMesh,xMesh,zMBCRsig)
hold on
plot3(xx(:,1),xx(:,2),y,'k.')

subplot(2,2,1)
title('True Function')
meshc(xMesh,xMesh,zTrue)


figure
hold on
plot(xList(:,1),xList(:,2),'ro')
plot(xList(:,3),xList(:,4),'gd')
plot(xList(:,5),xList(:,6),'bx')
plot(xTrue(1),xTrue(2),'k+')
legend('LSE','CAP','MBCR','TRUE')

% figure
% plot(xList(:,1),xList(:,2),'ro')
% 
% figure
% plot(xList(:,3),xList(:,4),'gd')
% 
% figure
% plot(xList(:,5),xList(:,6),'bx')



