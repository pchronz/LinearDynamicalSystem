function [mu, V, P] = computeForwardRecursion(x, A, Gamma, C, Sigma, V_m1, mu_m1, P_m1)
  D_z=length(mu_m1);
  K=P_m1*C'*inv(C*P_m1*C'+Sigma);
  mu=A*mu_m1+K*(x-C*A*mu_m1);
  V=(eye(D_z)-K*C)*P_m1;
  P=A*V*A'+Gamma;
endfunction
