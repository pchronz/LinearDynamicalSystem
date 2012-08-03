% TODO profile
% TODO initialize with something meaningful such as the result of a k-means run
% TODO particle filters
% TODO compare to other implementations
% TODO simulate covariance in prediction

% package dependencies
% statistics package for the MV normal PDF
pkg load statistics

% clear all current variables
clear

% observations
N=50;
R=250;
D_x=1;
D_z=2;

%X=randn(D_x, N);

for r=1:R
  X(:,:,r)=sin((linspace(0,2,N))*pi*1)+0.6*randn(D_x,N);
  for i=2:D_x
    X(i,:,r)=shift(X(i-1,:,r), 1);
  endfor
endfor

%X(:,:,1)=repmat([1,0], 1, N/2);
%X(:,:,2)=repmat([1,0], 1, N/2);

%X=repmat(linspace(0,1,N), D_x, 1);
%X=repmat(linspace(0,1,N).^2, D_x, 1);

[lds, muhats, likelihoods]=learnLDS(X, D_z, 400, 0.2);

% plot the likelihoods
subplot(2,2,1)
plot(likelihoods)

if(D_x == 1)
  % plot the first sequence
  subplot(2,2,2)
  plot(X(:,:,1))

  [X_pred, Z_pred]=predictLDS(lds, X(:,:,1), N);

  subplot(2,2,3)
  plot([X(:,:,1), X_pred])

  subplot(2,2,4)
  plot(Z_pred')
endif


