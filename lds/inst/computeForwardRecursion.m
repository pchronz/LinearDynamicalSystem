% Compute the distribution of the given latent variable given the observations from x_1 until x_n
% 
% Inputs
% x - data vector corresponding to the current latent variable
% A - linear transformation matrix used in p(x|z)
% Gamma - covariance matrix in p(z_n|z_n-1)
% C - linear transformation matrix used in p(z_n|z_n-1)
% Sigma - covariance matrix in p(x|z_n)
% V_m1 - covariance of the previous local marginal (i.e. cov(p(z_n-1))), as obtained in the previous step
% mu_m1 - mean of the previous local marginal (i.e. E[p(z_n-1)]), as obtained in the previous step
% P_m1 - value of P as obtained in the previous step
% 
% Outputs
% mu - mean of the distribution of the current latent variable
% V - covariance of the distribution of the current latent variable
% P - intermediate result to be used in following steps and for the backward recursions

function [mu, V, P] = computeForwardRecursion(x, A, Gamma, C, Sigma, V_m1, mu_m1, P_m1)
  % compute the distribution of the given latent variable given the observations from x_1 until x_n
  % p(z_n|x_1,...,x_n) according to (13.84)

  D_z=length(mu_m1);

  K=(P_m1*C')/(C*P_m1*C'+Sigma); % (13.92)
  mu=A*mu_m1+K*(x-C*A*mu_m1); % (13.89)
  V=(eye(D_z)-K*C)*P_m1; % (13.90)
  P=A*V*A'+Gamma; % (13.88)
endfunction

