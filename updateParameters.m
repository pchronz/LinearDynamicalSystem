function [mu0_new, P0_new, A_new, Gamma_new, C_new, Sigma_new] = updateParameters(muhats, Vhats, Js, X)
  N=length(X(1,:));
  D_x=length(X(:,1));
  D_z=length(muhats(:,1));

  % mu0_new
  mu0_new=expectZn(muhats(:,1));

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
endfunction
