function p = drawDirichlet(alpha)

% This function draws from a Dirichlet distribution with parameter alpha =
% (alpha_1,...,alpha_d).  Set alpha_1 = ... = alpha_d = 1 for a uniform
% distribution on the probability simplex.

d = length(alpha);
p = randg(alpha);
p = p/sum(p);