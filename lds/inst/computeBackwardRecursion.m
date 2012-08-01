% compute the distribution of the given latent variable given all observations p(x|X)
%
% Inputs
% mu - mean of the same latent variable obtained by the forward recursion E[alpha^(z_n)]
% V - covariance matrix corresponding to mu
% mu_hat_p1 - mean of the succeeding step of the backwards recursion E[gamma^(z_n+1)]
% V_hat_p1 - covariance matrix corresponding to mu_hat_p1
% A - linear transformation matrix in p(x|z)
% P - as obtained in the forward recursion
%
% Outputs
% mu_hat - mean value of the current latent variable given all observatoions E[gamma^(z_n)]
% V_hat - variance corresponding to mu_hat
% J - intermediate result required during the parameter update in the M-step of the EM algorithm

function [mu_hat, V_hat, J] = computeBackwardRecursion(mu, V, mu_hat_p1, V_hat_p1, A, P)
  % compute the distribution of the given latent variable given all observations
  % p(z|X) according to (13.98)

  J=(V*A')/P; % (13.102)
  mu_hat=mu+J*(mu_hat_p1-A*mu); % (13.100)
  V_hat=V+J*(V_hat_p1-P)*J'; % (13.101)
endfunction

