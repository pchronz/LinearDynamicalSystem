% TODO use bsxfn as much as possible
% TODO parallelize
% TODO document dimensions of matrices
% TODO document used formulas
% TODO create API
% TODO automatic testing via assertions and test cases

% package dependencies
% statistics package for the MV normal PDF
pkg load statistics

% clear all current variables
clear

% maximum iterations
MAX_ITER=150;

% length of the learned sequence
N=50;

% number of training sequences
R=10;

% dimensionality of the observed variables
D_x=1;
% dimensionality of the latent variables
D_z=2;

% observations
%X=randn(D_x, N);

X(:,:,1)=sin(linspace(0,1,N)*pi*1);
for i=2:D_x
  X(i,:,1)=shift(X(i-1,:,1), 1);
endfor
for r=2:R
  X(:,:,r)=X(:,:,r-1);
endfor

%X(:,:,1)=repmat([1,0], 1, N/2);
%X(:,:,2)=repmat([1,0], 1, N/2);

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
 while it<MAX_ITER && deltaLikelihoods > 0.0005
   % forward recursions (alpha values) for all sequences
   mus=zeros(D_z, N, R);
   Vs=zeros(D_z, D_z, N, R);
   Ps=zeros(D_z, D_z, N, R);
   for r=1:R
     [mus(:, :, r), Vs(:, :, :, r), Ps(:, :, :, r)]=computeForwardRecursions(P0, C, Sigma, mu0, X(:,:,r), A, Gamma);
   endfor

   % backward recursions (gamma values) for all sequences
   Js=zeros(D_z, D_z, N, R);
   muhats=zeros(D_z, N, R);
   Vhats=zeros(D_z, D_z, N, R);
   for r=1:R
     [muhats(:, :, r), Vhats(:, :, :, r), Js(:, :, :, r)]=computeBackwardRecursions(mus(:, :, r), Vs(:, :, :, r), Ps(:, :, :, r), A);
   endfor

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

if(D_x == 1)
  % plot the first sequence
  subplot(2,2,2)
  plot(X(:,:,1))

  if(R>1)
    % plot the second sequence
    subplot(2,2,3)
    plot(X(:,:,2))
  endif

  % do forward recursion until the last observation
  [mus, Vs, Ps]=computeForwardRecursions(P0, C, Sigma, mu0, X(:,:,1), A, Gamma);

  % compute and plot predictions
  % predict the next latent variable given current observations
  % extend the required matrices
  N_pred=100;
  X=[X(:,:,1),zeros(D_x,N_pred)];
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
  subplot(2,2,4)
  plot(X)
endif



