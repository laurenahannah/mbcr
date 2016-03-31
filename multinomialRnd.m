function outVec = multinomialRnd(n,p)
[d,r] = size(p);
outVec = zeros(size(p));
dd = max(d,r);
cs = cumsum(p);
for i = 1:n
    u = rand;
    j = 1;
    while (cs(j) < u)
        j = j+1;
    end
    outVec(j) = outVec(j) + 1;
end