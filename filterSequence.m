function [muhats, Vhats] = filterSequence(lds, X)
  [mus, Vs, Ps] = computeForwardRecursions(lds.P0, lds.C, lds.Sigma, lds.mu0, X, lds.A, lds.Gamma);
  [muhats, Vhats, Js] = computeBackwardRecursions(mus, Vs, Ps, lds.A);
endfunction


