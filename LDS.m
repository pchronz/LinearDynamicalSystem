% TODO profile
% TODO initialize with something meaningful such as the result of a k-means run
% TODO particle filters
% TODO abort when encountering multiple negative gradients in likelihoods

% package dependencies
% statistics package for the MV normal PDF
pkg load statistics

% clear all current variables
clear

% observations
N=50;
R=3;
D_x=1;
D_z=5;
%X=randn(D_x, N);

X(:,:,1)=sin(linspace(0,1,N)*pi*1);
for i=2:D_x
  X(i,:,1)=shift(X(i-1,:,1), 1);
endfor
for r=2:R
  X(:,:,r)=X(:,:,r-1);
endfor

%X(:,:,1)=repmat([1,0], 1, N/2);
%X(:,:,2)=repmat([1,0], 1, N/2);

%X=repmat(linspace(0,1,N), D_x, 1);
%X=repmat(linspace(0,1,N).^2, D_x, 1);

[lds, muhats, likelihoods]=learnLDS(X, D_z, 150, 0.001);

% plot the likelihoods
subplot(2,2,1)
plot(likelihoods)

if(D_x == 1)
  % plot the first sequence
  subplot(2,2,2)
  plot(X(:,:,1))

  if(R>1)
    % plot the second sequence
    subplot(2,2,3)
    plot(X(:,:,2))
  endif

  X_pred=predictLDS(lds, X(:,:,1), 100);

  subplot(2,2,4)
  plot([X(:,:,1), X_pred])
endif


