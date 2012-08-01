% compute the local marginals conditioned on the observations up to the repsective latent variable one by one
% 
% Inputs
% P0, C, Sigma, mu0, A, Gamma - model parameters
% X - a DxN matrix. D corresponds to the dimensionality of the data vectors,
% and N corresponds to the number of steps in the sequence X
% 
% Outputs
% mus - DxN matrix. D corresponds to the dimensionality of the latent space, N to the number of observations in the current sequence. The values correspond to the mean values of the alpha^ distributions
% Vs -  DxDxN covariance matrices corresponding to mus
% Ps - intermediate results, which can be reused in the backward recursions

function [mus, Vs, Ps] = computeForwardRecursions(P0, C, Sigma, mu0, X, A, Gamma)
  % compute the local marginals conditioned on the observations up to the repsective latent variable one by one
  % p(z_n|x_1,...,x_n) for all latent variables z_n according to (13.84, 13.88-13.92, 13.94-13.97)

  D_z=length(mu0);
  N=length(X(1,:));

  % allocate the matrices for the intermediate results
  Ps=zeros(D_z, D_z, N);
  mus=zeros(D_z, N);
  Vs=zeros(D_z, D_z, N);

  % perform computations of forward run and thus local marginals incl past observations (alpha values)
  % compute the first step with the given parameters (13.94-13.97)
  K=(P0*C')/(C*P0*C'+Sigma);
  mus(:,1)=mu0+K*(X(:,1)-C*mu0);
  Vs(:,:,1)=(eye(D_z)-K*C)*P0;
  Ps(:,:,1)=A*Vs(:,:,1)*A'+Gamma;

  % now for each step use the corresponding recursive formulae (13.88-13.91)
  for n=2:N
    [mus(:,n), Vs(:,:,n), Ps(:,:,n)] = computeForwardRecursion(X(:,n), A, Gamma, C, Sigma, Vs(:,:,n-1), mus(:,n-1), Ps(:,:,n-1));
  endfor
endfunction

