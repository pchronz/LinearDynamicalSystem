function [mu_hat, V_hat, J] = computeBackwardRecursion(mu, V, mu_hat_p1, V_hat_p1, A, P)
  J=(V*A')/P;
  mu_hat=mu+J*(mu_hat_p1-A*mu);
  V_hat=V+J*(V_hat_p1-P)*J';
endfunction
