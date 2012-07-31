function [mus, Vs, Ps] = computeForwardRecursions(P0, C, Sigma, mu0, X, A, Gamma)
  D_z=length(mu0);
  N=length(X(1,:));

  % allocate the matrices for the intermediate results
  Ps=zeros(D_z, D_z, N);
  mus=zeros(D_z, N);
  Vs=zeros(D_z, D_z, N);

  % perform computations of forward run and thus local marginals incl past observations (alpha values)
  % compute the first step with the given parameters (13.94-13.97)
  K=(P0*C')/(C*P0*C'+Sigma);
  mus(:,1)=mu0+K*(X(:,1)-C*mu0);
  Vs(:,:,1)=(eye(D_z)-K*C)*P0;
  Ps(:,:,1)=A*Vs(:,:,1)*A'+Gamma;
  % now for each step use the corresponding recursive formulas (13.88-13.91)
  for n=2:N
    [mus(:,n), Vs(:,:,n), Ps(:,:,n)] = computeForwardRecursion(X(:,n), A, Gamma, C, Sigma, Vs(:,:,n-1), mus(:,n-1), Ps(:,:,n-1));
  endfor
endfunction

