% This function will predict N_pred values based on the observations in X based on the model lds.
% 
% Inputs
% lds - Model learned by learnLDS
% X - DxNxR matrix. D is the dimensionality of the data, N the number of observations in each sequence, R the number of observations. The series in X will be used as basis for the predictions.
% N_pred - number of steps to predict based on X
% 
% Outputs
% X_preds - DxN_pred matrix containing the predictions x_n as column vectors for each prediction step.

function [X_pred, Z_pred] = simulateLDS(lds, X, N_pred)
  % TODO check for supplied arguments
  % load the statistics package for the MV normal PDF
  pkg load statistics

  D_x=length(X(:,1));
  D_z=length(lds.mu0);

  % do forward recursion until the last observation
  % this computes the alpha^ values, which are the posterior distribution for each
  % latent variable given observation up to the respective point
  [mus, Vs, Ps]=computeForwardRecursions(lds.P0, lds.C, lds.Sigma, lds.mu0, X, lds.A, lds.Gamma);

  % predict the next latent variable given current observations
  % the following can be dervied from (13.44)
  % note that (13.44) can only be used directly to compute p(z_N+1|X) and p(x_N+1|X)
  % for p(z_N+n|X) and p(x_N+n|X) for n=2,... the formulas below have to be used
  % basically just choose the most probable value (the mean) and then transform it
  % given (13.75) and (13.76) to obtain the next mean values
  X_pred=zeros(D_x, N_pred);
  % the predicted mean values
  mu_preds=zeros(D_z, N_pred);
  % the predicted covariance matrices
  U_preds=zeros(D_z, D_z, N_pred);
  % compute the mean of the next latent variable
  mu_preds(:,1)=lds.A*mus(:,end);
  % compute the covariance matrix of the next latent variable
  U_preds(:,:,1)=lds.Gamma+lds.A*Vs(:,:,end)*lds.A';
  for i=2:N_pred
    % compute the mean of the next latent variable
    mu_preds(:,i)=lds.A*mu_preds(:,i-1);
    % compute the covariance matrix of the next latent variable
    U_preds(:,:,i)=lds.Gamma+lds.A*U_preds(:,:,i-1)*lds.A';
  endfor

  % simulate the predictions
  for i=1:N_pred
    % draw a sample from the computed distribution
    X_pred(:,i)=mvnrnd((lds.C*mu_preds(:,i))', lds.Sigma+lds.C*U_preds(:,:,i)*lds.C')';
  end

  Z_pred=mu_preds;
endfunction

