% compute the local marginals conditioned on all observations one by one
%
% Input
% mus - means of local marginals as obtained by the forward recursion
% Vs - covariance matrices as obtained by the forward recursion
% Ps - intermediate results obtained by the forward recursion
% A - linear transformation matrix for p(x|z)
%
% Output
% muhats - mean values obtained by backward recursion
% Vhats - covariance matrices corresponding to muhats


function [muhats, Vhats, Js] = computeBackwardRecursions(mus, Vs, Ps, A)
  % compute the local marginals conditioned on all observations one by one
  % p(z|X) for all latent variables z according to (13.98, 13.100-13.102)
  D_z=length(mus(:,1));
  N=length(mus(1,:));

  % perform computations for backward run and thus local marginals incl also future observations (gamma values)
  Js=zeros(D_z, D_z, N);
  muhats=zeros(D_z, N);
  Vhats=zeros(D_z, D_z, N);
  
  % initializing the results for the last latent variable with the outcome of the forward pass
  % gamma^(z_N)=p(z_N|x_1,...,x_N)=p(z_N|X)=alpha^(z_N) => mu^_N=mu_N, V^_N=V_N
  muhats(:, N)=mus(:,N);
  Vhats(:,:,N)=Vs(:,:,N);
  Js(:,:,N)=(Vs(:,:,N)*A')/Ps(:,:,N);

  % starting at N-1 since gamma(z_N)=alpha(z_N) and thus the values for N are being initialized with the results for the last latent variable from the forward run
  for n=N-1:-1:1
    [muhats(:,n), Vhats(:,:,n), Js(:,:,n)]=computeBackwardRecursion(mus(:,n), Vs(:,:,n), muhats(:,n+1), Vhats(:,:,n+1), A, Ps(:,:,n));
  endfor
endfunction

