% TODO incorporate multiple independent sequences for learning
% TODO modularize into functions
% TODO parallelize
% TODO optimize: how to? cholesky decomposition for inverse of PD matrices?
% TODO replace zeros with NaNs?
% TODO adjust sizes of x and z to be independent
% TODO document used formulas
% TODO use scaling factors
% TODO predict next observed
% TODO predict next latent
% length of the prediction
N=10;

% observations
X=repmat([1,2,3]',1,100);

% initial settings
% TODO initialize with something meaningful such as the result of a k-means run
Sigma=eye(3);
C=ones(3);
A=ones(3);
Gamma=eye(3);
P0=eye(3);
mu0=ones(3, 1);

% learn the model parameters using EM
% TODO find a proper ending criterion
for it=1:1

  % 3D matrix containing the intermediate results
  Ps=zeros(3, 3, N);
  Ks=zeros(3, 3, N);
  mus=zeros(3, N);
  Vs=zeros(3, 3, N);

  % perform computations of forward run and thus local marginals incl past observations (alpha values)
  % TODO use maximized P0 and m0 according to EM
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
    Vs(:,:,n)=(eye(length(X(:,n)))-Ks(:,:,n)*C)*P;
  endfor
  Ps(:,:,N)=A*Vs(:,:,n-1)*A'+Gamma;

  % perform computations for backward run and thus local marginals incl also future observations (gamma values)
  Js=zeros(3, 3, N);
  muhats=zeros(3, N);
  Vhats=zeros(3, 3, N);
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
  P0_new=expectZnZn(Vhats(:,:,1),muhats(:,:,1))-expectZn(muhats(:,1))*expectZn(muhats(:,1))';

  % A_new
  % first sum over the expectations
  exp1=0;
  for n=2:N
    exp1=exp1+expectZnZnm1(Vhats(:,:,n), Js(:,:,n-1), muhats(:,n), muhats(:,n-1));
  endfor
  exp2=0;
  for(n=2:N)
    exp2=exp2+expectZnZn(Vhats(:,:,n-1),muhats(:,n-1));
  endfor
  A_new=exp1+inv(exp2);

  % Gamma_new
  G=zeros(length(X(:,1)));
  for n=2:N
    % TODO verify that the calculation of E[z_n-1, z_n] is correct
    G=G+expectZnZn(Vhats(:,:,n),muhats(:,n))-A_new*expectZnZnm1(Vhats(:,:,n-1),Js(:,:,n),muhats(:,n-1),muhats(:,n))-expectZnZnm1(Vhats(:,:,n),Js(:,:,n-1),muhats(:,n),muhats(:,n-1))+A_new*expectZnZn(Vhats(:,:,n-1),muhats(:,n-1))*A_new';
  endfor
  Gamma_new=1/(N-1)*G;

  % C_new
  exp1=0;
  for n=1:N
    exp1=exp1+X(:,n)*expectZn(muhats(:,n))';
  endfor
  exp2=0;
  for n=1:N
    exp2=exp2+expectZnZn(Vhats(:,:,n), muhats(:,n));
  endfor
  C_new=exp1*inv(exp2);

  % Sigma_new
  S=zeros(length(X(:,1)));
  for n=1:N
    S=S+X(:,n)*X(:,n)'-C_new'*expectZn(muhats(:,n))*X(:,n)'-X(:,n)*expectZn(muhats(:,n))'*C_new+C_new'*expectZnZn(Vhats(:,:,n),muhats(:,n))*C_new;
  endfor
  Sigma_new=1/N.*S;

  % assign the updated parameters
  mu0=mu0_new;
  P0=P0_new;
  A=A_new;
  Gamma=Gamma_new;
  C=C_new;
  Sigma=Sigma_new;

  % compute likelihood to monitor the progress
  % using the scaling factors to compute the likelihood according to (13.63): p(X)=c_1*c_2*...*c_N
  mu_c1=C*mu0;
  Sigma_c1=C*P0*C'+Sigma;
  D=length(X(:,1));
  likelihood=(2*pi)^(-D/2) * (1/sqrt(det(Sigma_c1))) * exp(-.5*(X(:,1)-mu_c1)'*inv(Sigma_c1)*(X(:,1)-mu_c1));
  for n=2:N
    % first compute the mean and covariance for cn
    mu_cn=C*A*mus(:,n-1);
    Sigma_cn=C*Ps(:,:,n-1)*C'+Sigma;

    % compute the probability for our observation from cn's distribution
    p_cn = (2*pi)^(-D/2) * (1/sqrt(det(Sigma_cn))) * exp(-.5*(X(:,n)-mu_cn)'*inv(Sigma_cn)*(X(:,n)-mu_cn));

    % multiply by previous result
    likelihood=likelihood*p_cn;
  endfor

  % print out the likelihood
  likelihood

endfor


% compute prediction on next latent variable

% compute prediction on next observation


