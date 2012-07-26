% TODO incorporate multiple independent sequences for learning
% TODO optimize: how to? cholesky decomposition for inverse of PD matrices?
% TODO parallelize
% TODO document used formulas
% TODO automatic testing via assertions and test cases
% TODO detect singularities and abort

% package dependencies
% statistics package for the MV normal PDF
pkg load statistics

% clear all current variables
clear

% maximum iterations
MAX_ITER=1500;

% length of the prediction
N=100;

% dimensionality of the observed variables
D_x=1;
% dimensionality of the latent variables
D_z=5;

% observations
X=randn(D_x, N);
%X=sin(linspace(0,2,N)*pi*1);
%for(i=2:D_x)
%  X(i,:)=shift(X(i-1,:), 1);
%endfor
%X=repmat([1,0], 1, N/2);
%X=repmat(linspace(0,1,N), D_x, 1);

% initial settings
% TODO initialize with something meaningful such as the result of a k-means run
Sigma=eye(D_x);
C=ones(D_x, D_z)./(D_x*D_z);
A=ones(D_z)./(D_z^2);
Gamma=eye(D_z);
P0=eye(D_z);
mu0=ones(D_z, 1)./D_z;

% learn the model parameters using EM
likelihood=zeros(MAX_ITER,1);
% TODO find a proper ending criterion
for it=1:MAX_ITER
  % allocate the matrices for the intermediate results
  Ps=zeros(D_z, D_z, N);
  mus=zeros(D_z, N);
  Vs=zeros(D_z, D_z, N);

  % perform computations of forward run and thus local marginals incl past observations (alpha values)
  % compute the first step with the given parameters (13.94-13.97)
  K=P0*C'*inv(C*P0*C'+Sigma);
  mus(:,1)=mu0+K*(X(:,1)-C*mu0);
  Vs(:,:,1)=(eye(D_z)-K*C)*P0;
  Ps(:,:,1)=A*Vs(:,:,1)*A'+Gamma;
  % now for each step use the corresponding recursive formulas (13.88-13.91)
  for n=2:N
    [mus(:,n), Vs(:,:,n), Ps(:,:,n)] = computeForwardRecursion(X(:,n), A, Gamma, C, Sigma, Vs(:,:,n-1), mus(:,n-1), Ps(:,:,n-1));
  endfor

  % perform computations for backward run and thus local marginals incl also future observations (gamma values)
  Js=zeros(D_z, D_z, N);
  muhats=zeros(D_z, N);
  Vhats=zeros(D_z, D_z, N);
  % initializing the results for the last latent variable with the outcome of the forward pass
  muhats(:, N)=mus(:,N);
  Vhats(:,:,N)=Vs(:,:,N);
  Js(:,:,N)=Vs(:,:,N)*A'*inv(Ps(:,:,N));
  % starting at N-1 since gamma(z_N)=alpha(z_N) and thus the values for N are being initialized with the results for the last latent variable from the forward run
  for n=N-1:-1:1
    [muhats(:,n), Vhats(:,:,n), Js(:,:,n)]=computeBackwardRecursion(mus(:,n), Vs(:,:,n), muhats(:,n+1), Vhats(:,:,n+1), A, Ps(:,:,n));
  endfor

  % compute log-likelihood to monitor the progress
  % save the likelihood for plotting
  likelihoods(it)=computeLogLikelihood(mu0, P0, C, Sigma, A, mus, Ps, X);

  % update the parameters 
  [mu0, P0, A, Gamma, C, Sigma]=updateParameters(muhats, Vhats, Js, X);
endfor

% plot the likelihoods
subplot(2,2,1)
plot(likelihoods)

% plot the data
subplot(2,2,2)
plot(X)

if(D_x == 1)
  % TODO compute the proper probability distribution instead of just using the mean?
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




