% Update the model parameters given the local marginals. This function will be called at each step of the EM algorithm after the local marginals (gamma^(z)) have been evaluated. This function corresponds to the M-step and maximizes the model parameters to maximize the complete data log-likelihood given the posterior using the old parameters values.
% 
% Inputs
% muhats - mean values of the local marginals of the latent variables given all observations.
% Vhats - covariance matrices corresponding to muhats.
% Js - intermeiate results obtained during the backward recursions.
% X - DxNxR matrix. D is the dimensionality of the data, N the number of observations in each sequence, R the number of observations.
% 
% Outputs
% mu0_new, P0_new, A_new, Gamma_new, C_new, Sigma_new - updated (maximized) model parameters.

function [mu0_new, P0_new, A_new, Gamma_new, C_new, Sigma_new] = updateParameters(muhats, Vhats, Js, X)
  % update the model parameters given the local marginals

  R=length(X(1,1,:));
  N=length(X(1,:,1));
  D_x=length(X(:,1,1));
  D_z=length(muhats(:,1,1));

  % mu0_new, (13.110) extended for multiple sequences similar to Ex. 13.12
  mu0_new=zeros(D_z,1);
  for r=1:R
    mu0_new=mu0_new+expectZn(muhats(:,1,r)); 
  endfor
  mu0_new=(1/R).*mu0_new;

  % P0_new, (13.111) extended for multiple sequences similar to Ex. 13.12
  P0_new=zeros(D_z, D_z);
  for r=1:R
    P0_new=P0_new+expectZnZn(Vhats(:,:,1,r),muhats(:,1,r));
  endfor
  P0_new=(1/R).*P0_new-mu0_new*mu0_new';

  % A_new, (13.113) extended for multiple sequences similar to Ex. 13.12
  % first sum over the expectations
  exp1=zeros(D_z);
  for r=1:R
    for n=2:N
      exp1=exp1+expectZnZnm1(Vhats(:,:,n,r),Js(:,:,n-1,r),muhats(:,n,r),muhats(:,n-1,r));
    endfor
  endfor
  exp2=zeros(D_z);
  for r=1:R
    for n=2:N
      exp2=exp2+expectZnZn(Vhats(:,:,n-1,r),muhats(:,n-1,r));
    endfor
  endfor
  A_new=exp1/exp2;

  % Gamma_new, (13.114) extended for multiple sequences similar to Ex. 13.12
  G=zeros(D_z);
  for r=1:R
    for n=2:N
      G=G+expectZnZn(Vhats(:,:,n,r),muhats(:,n,r))-A_new*expectZnm1Zn(Vhats(:,:,n,r),Js(:,:,n-1,r),muhats(:,n,r),muhats(:,n-1,r))-expectZnZnm1(Vhats(:,:,n,r),Js(:,:,n-1,r),muhats(:,n,r),muhats(:,n-1,r))*A_new'+A_new*expectZnZn(Vhats(:,:,n-1,r),muhats(:,n-1,r))*A_new';
    endfor
  endfor
  Gamma_new=(1/(N-1))*(1/R)*G;

  % C_new, (13.115) extended for multiple sequences similar to Ex. 13.12
  exp1=zeros(D_x, D_z);
  for r=1:R
    for n=1:N
      exp1=exp1+X(:,n,r)*expectZn(muhats(:,n,r))';
    endfor
  endfor
  exp2=zeros(D_z);
  for r=1:R
    for n=1:N
      exp2=exp2+expectZnZn(Vhats(:,:,n,r), muhats(:,n,r));
    endfor
  endfor
  C_new=exp1/exp2;

  % Sigma_new, (13.116) extended for multiple sequences similar to Ex. 13.12
  S=zeros(D_x);
  for r=1:R
    for n=1:N
      S=S+X(:,n,r)*X(:,n,r)'-C_new*expectZn(muhats(:,n,r))*X(:,n,r)'-X(:,n,r)*expectZn(muhats(:,n,r))'*C_new'+C_new*expectZnZn(Vhats(:,:,n,r),muhats(:,n,r))*C_new';
    endfor
  endfor
  Sigma_new=(1/N)*(1/R).*S;
endfunction


function expZn = expectZn(muhat)
  % compute the expectation of z_n under the posterior probabilty
  % for Z given the current parameters p(Z|X)
  expZn=muhat; % (13.105)
endfunction

function expZnZn = expectZnZn(Vhat, muhat)
  % compute the expectation E[Z_n*Z_n] given the
  % posterior probability p(Z|X)
  expZnZn=Vhat+muhat*muhat'; % (13.107)
endfunction

function expZnZnm1 = expectZnZnm1(Vhat, Jn_m1, muhat, muhat_m1)
  % compute the expectation E[Z_n*Z_n-1] given the
  % posterior probability p(Z|X)
  expZnZnm1=Vhat*Jn_m1'+muhat*muhat_m1'; % (13.106)
endfunction

function expZnm1Zn = expectZnm1Zn(Vhat, Jn_m1, muhat, muhat_m1)
  % compute the expectation E[Z_n-1*Z_n] given the
  % posterior probability p(Z|X)

  % can be dervied from (13.104) and the definition of the covariance matrix (1.42)
  expZnm1Zn=Jn_m1*Vhat+muhat_m1*muhat'; 
endfunction




