function y = fMBCR(x,abStruct)
nSamp = length(abStruct);
y = 0;
[n,d] = size(x);
if (n > 1)
    x = x';
end
for i = 1:nSamp
    alpha = abStruct(i).alpha;
    beta = abStruct(i).beta;
    y = y + 1/nSamp*max([1,x]*[alpha;beta]);
end