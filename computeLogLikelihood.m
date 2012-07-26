function logLikelihood = computeLogLikelihood(mu0, P0, C, Sigma, A, mus, Ps, X)
  N=length(X(1,:));
  % using the scaling factors to compute the likelihood according to (13.63): p(X)=c_1*c_2*...*c_N
  mu_c1=C*mu0;
  Sigma_c1=C*P0*C'+Sigma;
  likelihood=mvnpdf(X(:,1)', mu_c1', Sigma_c1);
  for n=2:N
    % first compute the mean and covariance for cn
    mu_cn=C*A*mus(:,n-1);
    Sigma_cn=C*Ps(:,:,n-1)*C'+Sigma;

    % compute the probability for our observation from cn's distribution
    p_cn=mvnpdf(X(:,n)', mu_cn', Sigma_cn);

    % multiply by previous result
    likelihood=likelihood+log(p_cn);
  endfor
  logLikelihood=likelihood;
endfunction
