% TODO find proper ending criterion based on N, R, ...
% TODO profile
% TODO compare to other implementations
% TODO particle filters
% TODO examples/tutorial 
% TODO error messages/error handling
% TODO improve documentation
% TODO align with Octave OSS requirements
% TODO license
% TODO write a paper and publish
% TODO publish

% package dependencies
% statistics package for the MV normal PDF
pkg load statistics

% clear all current variables
clear

% observations
N=15;
R=25;
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

disp('Starting with the learning')
tic()
[lds, muhats, likelihoods]=learnLDS(X, D_z, 50, 0.0001);
toc()

% plot the likelihoods
subplot(3,2,1)
plot(likelihoods)

if(D_x == 1)
  % plot the first sequence
  subplot(3,2,2)
  plot(X(:,:,1))

  disp('Prediction...')
  tic()
  [X_pred, Z_pred]=predictLDS(lds, X(:,:,1), N);
  toc()

  subplot(3,2,3)
  plot([X(:,:,1), X_pred])

  subplot(3,2,4)
  plot(Z_pred')

  % compute the most probable sequence
  disp('Filtering')
  tic()
  [muhats, Vhats] = filterSequence(lds, X(:, :, 1));
  toc()
  subplot(3, 2, 5)
  plot(muhats')

endif


