# length of the prediction
N=10;

# observations
x=linspace(1, 100)';

# initial settings
Sigma=ones(3)+eye(3);
C=ones(3);
A=ones(3);
Gamma=ones(3)+eye(3);

# learn the model parameters using EM

# 3D matrix containing the intermediate results
Ps=zeros(3, 3, N);
Ks=zeros(3, 3, N);
mus=zeros(3, N);
Vs=zeros(3, 3, N);

# perform computations of forward run and thus local marginals incl past observations (alpha values)
P0=ones(3)+eye(3);
mu0=ones(3, 1);
for n=1:N
  if(n==1)
    P=P0;
    mu=mu0;
  else
    P(:,:,n-1)=A*Vs(:,:,n-1)*A'+Gamma;
    P=Ps(:,:,n-1);
    mu=mus(:,n-1);
  endif

  Ks(:,:,n)=P*C'*inv(C*P*C'+Sigma);
  mus(:,n)=A*mu+Ks(:,:,n)*(x(n)-C*A*mu);
  Vs(:,:,n)=(I-Ks(:,:,n)*C)*P;
endfor

# perform computations for backward run and thus local marginals incl also future observations (gamma values)
Js=zeros(3, 3, N)
muhats=zeros(3, N+1)
Vhats=zeros(3, 3, N+1)
# XXX are the followin initializations right?
muhats=ones(3, 1)
Vhats(:,:,N)=eye(3)
for n=N:-1:1
  Js(:,:,n)=Vs(:,:,n)*A'*inv(Ps(:,:,n))
  muhats(:, n)=mus(:, n) + Js(:,:,n)*(muhats(:,n+1)-A*mus(:,n))
  Vhats(:,:,n)=Vs(:,:,n) + Js(:,:,n)*(Vhats(:,:,n+1)-Ps(:,:,n))*Js(:,:,n)'
endfor

# compute likelihood to monitor the progress

# compute prediction on next latent variable

# compute prediction on next observation


