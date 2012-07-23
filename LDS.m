% TODO incorporate multiple independent sequences for learning
% TODO modularize into functions
% TODO parallelize
% TODO optimize: how to? cholesky decomposition for inverse of PD matrices?
% TODO replace zeros with NaNs?
% TODO adjust sizes of x and z to be independent
% TODO document used formulas
% TODO predict next observed
% TODO predict next latent

% maximum iterations
MAX_ITER=150;

% length of the prediction
N=2;

% dimensionality of the observed variables
D_x=2;
% dimensionality of the latent variables
D_z=2;

% observations
%X=repmat([1,2,3]',1,100);
X=repmat([1,0],1,N/2);
for(i=2:D_x)
  X(i,:)=shift(X(i-1,:), 1);
endfor

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
  Ks=zeros(D_z, D_x, N);
  mus=zeros(D_z, N);
  Vs=zeros(D_z, D_z, N);

  % perform computations of forward run and thus local marginals incl past observations (alpha values)
  for n=1:N
    if(n==1)
      P=P0;
      mu=mu0;
    else
      Ps(:,:,n-1)=A*Vs(:,:,n-1)*A'+Gamma;
      P=Ps(:,:,n-1);
      mu=mus(:,n-1);
    endif

    Ks(:,:,n)=P*C'*inv(C*P*C'+Sigma);
    mus(:,n)=A*mu+Ks(:,:,n)*(X(:,n)-C*A*mu);
    Vs(:,:,n)=(eye(D_z)-Ks(:,:,n)*C)*P;
  endfor
  Ps(:,:,N)=A*Vs(:,:,n-1)*A'+Gamma;

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
    Js(:,:,n)=Vs(:,:,n)*A'*inv(Ps(:,:,n));
    muhats(:, n)=mus(:, n) + Js(:,:,n)*(muhats(:,n+1)-A*mus(:,n));
    Vhats(:,:,n)=Vs(:,:,n) + Js(:,:,n)*(Vhats(:,:,n+1)-Ps(:,:,n))*Js(:,:,n)';
  endfor

  % update the parameters 
  % mu0_new
  mu0_new=expectZn(muhats(:,n));

  % P0_new
  P0_new=expectZnZn(Vhats(:,:,1),muhats(:,1))-expectZn(muhats(:,1))*expectZn(muhats(:,1))';

  % A_new
  % first sum over the expectations
  exp1=zeros(D_z);
  for n=2:N
    exp1=exp1+expectZnZnm1(Vhats(:,:,n),Js(:,:,n-1),muhats(:,n),muhats(:,n-1));
  endfor
  exp2=zeros(D_z);
  for n=2:N
    exp2=exp2+expectZnZn(Vhats(:,:,n-1),muhats(:,n-1));
  endfor
  A_new=exp1*inv(exp2);

  % Gamma_new
  G=zeros(D_z);
  for n=2:N
    G=G+expectZnZn(Vhats(:,:,n),muhats(:,n))-A_new*expectZnm1Zn(Vhats(:,:,n),Js(:,:,n-1),muhats(:,n),muhats(:,n-1))-expectZnZnm1(Vhats(:,:,n),Js(:,:,n-1),muhats(:,n),muhats(:,n-1))*A_new'+A_new*expectZnZn(Vhats(:,:,n-1),muhats(:,n-1))*A_new';
  endfor
  Gamma_new=1/(N-1)*G;

  % C_new
  exp1=zeros(D_x, D_z);
  for n=1:N
    exp1=exp1+X(:,n)*expectZn(muhats(:,n))';
  endfor
  exp2=zeros(D_z);
  for n=1:N
    exp2=exp2+expectZnZn(Vhats(:,:,n), muhats(:,n));
  endfor
  C_new=exp1*inv(exp2);

  % Sigma_new
  S=zeros(D_x);
  for n=1:N
    S=S+X(:,n)*X(:,n)'-C_new*expectZn(muhats(:,n))*X(:,n)'-X(:,n)*expectZn(muhats(:,n))'*C_new'+C_new*expectZnZn(Vhats(:,:,n),muhats(:,n))*C_new';
  endfor
  Sigma_new=1/N.*S;

  % assign the updated parameters
  mu0=mu0_new;
  P0=P0_new;
  A=A_new;
  Gamma=Gamma_new;
  C=C_new;
  Sigma=Sigma_new;

  %% compute likelihood to monitor the progress
  %% using the scaling factors to compute the likelihood according to (13.63): p(X)=c_1*c_2*...*c_N
  %mu_c1=C*mu0;
  %Sigma_c1=C*P0*C'+Sigma;
  %likelihood=(2*pi)^(-D_x/2) * (1/sqrt(det(Sigma_c1))) * exp(-.5*(X(:,1)-mu_c1)'*inv(Sigma_c1)*(X(:,1)-mu_c1));
  %for n=2:N
  %  % first compute the mean and covariance for cn
  %  mu_cn=C*A*mus(:,n-1);
  %  Sigma_cn=C*Ps(:,:,n-1)*C'+Sigma;

  %  % compute the probability for our observation from cn's distribution
  %  p_cn = (2*pi)^(-D_x/2) * (1/sqrt(det(Sigma_cn))) * exp(-.5*(X(:,n)-mu_cn)'*inv(Sigma_cn)*(X(:,n)-mu_cn));

  %  % multiply by previous result
  %  likelihood=likelihood*p_cn;
  %endfor

  %% save the likelihood for plotting
  %likelihoods(it)=likelihood;

endfor

% plot the likelihoods
plot(likelihoods)


% compute prediction on next latent variable

% compute prediction on next observation

