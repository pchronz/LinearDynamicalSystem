% TODO incorporate multiple independent sequences for learning
% TODO parallelize
% TODO document used formulas
% TODO automatic testing via assertions and test cases

% package dependencies
% statistics package for the MV normal PDF
pkg load statistics

% clear all current variables
clear

% maximum iterations
MAX_ITER=1200;

% length of the learned sequence
N=100;

% dimensionality of the observed variables
D_x=1;
% dimensionality of the latent variables
D_z=1;

% observations
%X=randn(D_x, N);
X=sin(linspace(0,1,N)*pi*1);
for(i=2:D_x)
  X(i,:)=shift(X(i-1,:), 1);
endfor
%X=repmat([1,0], 1, N/2);
%X=repmat(linspace(0,1,N), D_x, 1);
%X=repmat(linspace(0,1,N).^2, D_x, 1);

% initial settings
% TODO initialize with something meaningful such as the result of a k-means run
Sigma=eye(D_x);
C=ones(D_x, D_z)./(D_x*D_z);
A=ones(D_z)./(D_z^2);
Gamma=eye(D_z);
P0=eye(D_z);
mu0=ones(D_z, 1)./D_z;

% learn the model parameters using EM
likelihoods=zeros(MAX_ITER,1);
deltaLikelihoods=1000;
try
  it=1;
  while it<MAX_ITER && deltaLikelihoods > 0.002
    % forward recursions (alpha values)
    [mus, Vs, Ps]=computeForwardRecursions(P0, C, Sigma, mu0, X, A, Gamma);

    % backward recursions (gamma values)
    [muhats, Vhats, Js]=computeBackwardRecursions(mus, Vs, Ps, A);

    % compute log-likelihood to monitor the progress
    % save the likelihood for plotting
    likelihoods(it)=computeLogLikelihood(mu0, P0, C, Sigma, A, mus, Ps, X);

    % update the parameters 
    [mu0, P0, A, Gamma, C, Sigma]=updateParameters(muhats, Vhats, Js, X);

    % ending criterions
    if(it>1)
      deltaLikelihoods=(likelihoods(it)-likelihoods(it-1))/abs(likelihoods(it));
    endif
    it=it+1;
  endwhile
catch
  disp(strcat("warning: EM algorithm ended due to an error (singularity?) after ", mat2str(it), " steps"))
  disp(strcat("relative increase in the likelihood: ", mat2str(deltaLikelihoods)))
end_try_catch

% plot the likelihoods
subplot(2,2,1)
plot(likelihoods)

% plot the data
subplot(2,2,2)
plot(X)

if(D_x == 1)
  % do forward recursion until the last observation
  [mus, Vs, Ps]=computeForwardRecursions(P0, C, Sigma, mu0, X, A, Gamma);

  % compute and plot predictions
  % predict the next latent variable given current observations
  % extend the required matrices
  N_pred=100;
  X=[X,zeros(D_x,N_pred)];
  mu_preds=zeros(D_z, N_pred);
  % compute the mean of the next latent variable
  mu_preds(:,1)=A*mus(:,N);
  % compute the most probable observation
  X(:,N+1)=C*mu_preds(:,1);
  for i=2:N_pred
    % compute the mean of the next latent variable
    mu_preds(:,i)=A*mu_preds(:,i-1);
    % compute the most probable observation
    X(:,N+i)=C*mu_preds(:,i);
  endfor
  subplot(2,2,3)
  plot(X)
endif

if(D_z == 1)
  subplot(2,2,4)
  plot([muhats, mu_preds])
endif




