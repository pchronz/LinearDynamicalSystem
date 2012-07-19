# length of the prediction
N=10;

# observations
x=linspace(1, 100)';

# initial settings
Sigma=ones(3)+eye(3);
C=ones(3);
A=ones(3);
Gamma=ones(3)+eye(3);

# cell array for the intermediate results
Ps=zeros(3, 3, N);
Ks=zeros(3, 3, N);
Vs=zeros(3, 3, N);

# initialization
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


