% Learn the model parameters for the linear dynamical system by maximizing the likelihood using the EM algorithm
% 
% Inputs
% X - DxNxR matrix. D is the dimensionality of the data, N the number of observations in each sequence, R the number of observations.
% D_z - dimensionality of the latent space
% maxIterations - maximal number of iterations to run the expectation maximization algorithm for. The EM algorithm will terminate under the following circumstances:
%   - a singularity appears due to the limited numerical precision while solving a linear system
%   - the EM algorithm decreases the log-likelihood due to the limiteted numerical precision of the system
%   - maxIterations iterations have been performed
%   - the relative increase in deltaLikelihoods is less than deltaLikelihoods (this implies ending criterion 2 if deltaLikelihoods > 0)
% minImprovement - ending criterion of the learning algorithm. The EM algorithm will abort if the relative increase in the log-likelihood is less than deltaLikelihoods
% 
% params - model parameters as optimized by the EM algorithm
% muhats - mean values obtained through the backward recursion. These can be used to compute the most probable sequence in the latent space given the observations X. This corresponds to the solution found by the Viterbi/max-sum algorithm in HMMs. 
% likelihoods - the values of the log-likelihood for each step of the EM algorithm. This provides information on the progress of the optimization.

function [params, muhats, likelihoods] = learnLDS(X, D_z, maxIterations, minImprovement)
  % TODO check for supplied arguments
  % return the model parameters for the LDS
  % also return muhats, since these represent the most probable sequence of latent variables

  % length of the learned sequence
  N=length(X(1,:,1));

  % number of training sequences
  R=length(X(1,1,:));

  % dimensionality of the observed variables
  D_x=length(X(:,1,1));

  % dimenstionality of the latent space is given by D_z

  % initial settings
  Sigma=eye(D_x)+ones(D_x); % covariance matrix for p(x|z) (13.76)
  C=eye(D_x, D_z); % linear transformation matrix for p(x|z) (13.76)
  A=eye(D_z); % linear transformation matrix for p(z_n|z_n-1) (13.75)
  Gamma=eye(D_z)+ones(D_z); % covariance matrix for p(z_n|z_n-1) (13.75)
  P0=eye(D_z)+ones(D_z)*1000; % model parameter
  mu0=zeros(D_z, 1)./D_z; % model parameter

  % backup the previous parameters in case we encounter a singularity
  Sigma_old=Sigma;
  C_old=C;
  A_old=A;
  Gamma_old=Gamma;
  P0_old=P0;
  mu0_old=mu0;

  % learn the model parameters using EM: PRML, 13.3.2
  likelihoods=zeros(maxIterations,1);
  deltaLikelihoods=1000;
  try
   it=1;
   while it<maxIterations && deltaLikelihoods > minImprovement
     % forward recursions (alpha values) for all sequences
     % this will compute the scaled conditional probablities 
     % for the latent variable given observations up to n
     % p(z_n|x_1,...,x_n) (13.84)
     mus=zeros(D_z, N, R);
     Vs=zeros(D_z, D_z, N, R);
     Ps=zeros(D_z, D_z, N, R);
     for r=1:R
       [mus(:, :, r), Vs(:, :, :, r), Ps(:, :, :, r)]=computeForwardRecursions(P0, C, Sigma, mu0, X(:,:,r), A, Gamma);
     endfor

     % backward recursions (gamma values) for all sequences
     % this will compute the scaled conditional probablities 
     % for the latent variable given *all* observations
     % p(z_n|x_1,...,x_N)=p(z_n|X) (13.98)
     Js=zeros(D_z, D_z, N, R);
     muhats=zeros(D_z, N, R);
     Vhats=zeros(D_z, D_z, N, R);
     for r=1:R
       [muhats(:, :, r), Vhats(:, :, :, r), Js(:, :, :, r)]=computeBackwardRecursions(mus(:, :, r), Vs(:, :, :, r), Ps(:, :, :, r), A);
     endfor

     % compute log-likelihood to monitor the progress
     % save the likelihood for plotting
     likelihoods(it)=computeLogLikelihood(mu0, P0, C, Sigma, A, mus, Ps, X);

     % backup the current parameter estimate
     mu0_old=mu0;
     P0_old=P0;
     A_old=A;
     Gamma_old=Gamma;
     C_old=C;
     Sigma_old=Sigma;

     % update the parameters 
     [mu0, P0, A, Gamma, C, Sigma]=updateParameters(muhats, Vhats, Js, X);

     % ending criteria
     if(it>1)
       deltaLikelihoods=(likelihoods(it)-likelihoods(it-1))/abs(likelihoods(it));
     endif

     % output the current iteration
     disp(strcat("Just finished iteration: ", mat2str(it)))
     it=it+1;
   endwhile
  catch
    % recover the parameter estimates from the previous iteration
    mu0=mu0_old;
    P0=P0_old;
    A=A_old;
    Gamma=Gamma_old;
    C=C_old;
    Sigma=Sigma_old;
    
    % issue warnings
    disp(strcat("warning: EM algorithm ended due to an error (singularity?) after ", mat2str(it), " steps"))
    disp(strcat("relative increase in the likelihood: ", mat2str(deltaLikelihoods)))
    disp(lasterror().message)
  end_try_catch

  if(deltaLikelihoods < minImprovement)
    disp(strcat("Terminated algorithm since the improvement slowed down: ", mat2str(deltaLikelihoods)))
  endif
  if(it == maxIterations)
    disp("MaxIterations has been reached")
  endif

  % return the learned model parameters
  % these parameters describe the LDS model fully
  % according to (13.75 - 13.77)
  params.mu0=mu0;
  params.P0=P0;
  params.A=A;
  params.Gamma=Gamma;
  params.C=C;
  params.Sigma=Sigma;

  if(nargout==1)
    varargout{1}=params;
  elseif(nargout==2)
    varargout{1}=params;
    % return the means of the local marginals p(z|X)
    varargout{2}=muhats;
  else
    varargout{1}=params;
    varargout{2}=muhats;
    % return the likelihood for evaluation of the optimization progress
    varargout{3}=likelihoods;
  endif
endfunction

