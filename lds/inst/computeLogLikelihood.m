% Compute the log likelihood of the data in X using the given model parameters
% 
% Inputs
% mu0, P0, C, Sigma, A, mus, Ps are the model parameters and are obtained using the learning function
% X is a DxN matrix. D corresponds to the dimensionality of the data vectors and N corresponds to the number of steps in the sequence X
% 
% Output
% logLikelihood - the log-likelihood for X given the provided model parameters

function logLikelihood = computeLogLikelihood(mu0, P0, C, Sigma, A, mus, Ps, X)
  % compute the log likelihood of the data in X using the given model parameters
  % this is done by multiplying the distribution of the scaling coefficients c_n (13.91)
  % for all n=1,...N according to (13.63)
  logLikelihood=0;
  R=length(X(1,1,:));
  N=length(X(1,:,1));

  % using the scaling factors to compute the likelihood according to (13.63): p(X)=c_1*c_2*...*c_N
  mu_c1=C*mu0;
  Sigma_c1=C*P0*C'+Sigma;
  for r=1:R
    likelihood=mvnpdf(X(:,1,r)', mu_c1', Sigma_c1);
    for n=2:N
      % first compute the mean and covariance for cn
      mu_cn=C*A*mus(:,n-1,r);
      Sigma_cn=C*Ps(:,:,n-1,r)*C'+Sigma;

      % compute the probability for our observation from cn's distribution
      p_cn=mvnpdf(X(:,n,r)', mu_cn', Sigma_cn);

      % add to previous result
      likelihood=likelihood+log(p_cn);
    endfor
    logLikelihood=logLikelihood+likelihood;
  endfor
endfunction

