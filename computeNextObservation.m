function x = computeNextObservation(A, mu, Gamma, V, C, Sigma)
  % compute prediction on next latent variable
  mu_znp1=A*mu;
  Gamma_znp1=Gamma+A*V*A';

  % compute prediction on next observation
  mu_xnp1=C*A*mu
  Sigma_xnp1=Sigma+C*(Gamma+A*V*C');
endfunction
