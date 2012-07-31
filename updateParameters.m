function [mu0_new, P0_new, A_new, Gamma_new, C_new, Sigma_new] = updateParameters(muhats, Vhats, Js, X)
  R=length(X(1,1,:));
  N=length(X(1,:,1));
  D_x=length(X(:,1,1));
  D_z=length(muhats(:,1,1));

  % mu0_new
  mu0_new=zeros(D_z,1);
  for r=1:R
    mu0_new=mu0_new+expectZn(muhats(:,1,r));
  endfor
  mu0_new=(1/R).*mu0_new;

  % P0_new
  P0_new=zeros(D_z, D_z);
  for r=1:R
    P0_new=P0_new+expectZnZn(Vhats(:,:,1,r),muhats(:,1,r))-expectZn(muhats(:,1,r))*expectZn(muhats(:,1,r))';
  endfor
  P0_new=(1/R).*P0_new;

  % A_new
  % first sum over the expectations
  exp1=zeros(D_z);
  for r=1:R
    for n=2:N
      exp1=exp1+expectZnZnm1(Vhats(:,:,n,r),Js(:,:,n-1,r),muhats(:,n,r),muhats(:,n-1,r));
    endfor
  endfor
  exp2=zeros(D_z);
  for r=1:R
    for n=2:N
      exp2=exp2+expectZnZn(Vhats(:,:,n-1,r),muhats(:,n-1,r));
    endfor
  endfor
  A_new=exp1/exp2;

  % Gamma_new
  G=zeros(D_z);
  for r=1:R
    for n=2:N
      G=G+expectZnZn(Vhats(:,:,n,r),muhats(:,n,r))-A_new*expectZnm1Zn(Vhats(:,:,n,r),Js(:,:,n-1,r),muhats(:,n,r),muhats(:,n-1,r))-expectZnZnm1(Vhats(:,:,n,r),Js(:,:,n-1,r),muhats(:,n,r),muhats(:,n-1,r))*A_new'+A_new*expectZnZn(Vhats(:,:,n-1,r),muhats(:,n-1,r))*A_new';
    endfor
  endfor
  Gamma_new=(1/(N-1))*(1/R)*G;

  % C_new
  exp1=zeros(D_x, D_z);
  for r=1:R
    for n=1:N
      exp1=exp1+X(:,n,r)*expectZn(muhats(:,n,r))';
    endfor
  endfor
  exp2=zeros(D_z);
  for r=1:R
    for n=1:N
      exp2=exp2+expectZnZn(Vhats(:,:,n,r), muhats(:,n,r));
    endfor
  endfor
  C_new=exp1/exp2;

  % Sigma_new
  S=zeros(D_x);
  for r=1:R
    for n=1:N
      S=S+X(:,n,r)*X(:,n,r)'-C_new*expectZn(muhats(:,n,r))*X(:,n,r)'-X(:,n,r)*expectZn(muhats(:,n,r))'*C_new'+C_new*expectZnZn(Vhats(:,:,n,r),muhats(:,n,r))*C_new';
    endfor
  endfor
  Sigma_new=(1/N)*(1/R).*S;
endfunction

