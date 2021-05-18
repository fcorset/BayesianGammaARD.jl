function Q=StaLawFunc(X,p,FT,FD,Pi,N)
% function Q=StaLawFunc(X,p,FT,FD,Pi,N)

% This function compute the stationary law for the predictive maintenance
% strategy considering improving RUL prognostic accuracy

% X is the [N x 1] vector of considered degradation levels
% N is the length of X
% Pi is the [N x 1] vector of stationary law at the lastest iteration.
% FT is the [N x N] matrix transition law when time interva = inspection period
% FD is the [N x N] matrix transition law when time interva = waiting time
% Q is the [N x 1] vector of stationary law at the actual iteration.

% © 17/05/2015 Tuan Huynh - tuan.huynh@utt.fr - UTT, France 

% %%%%%%%%%%% Compute indices vectors %%%%%%%%%%%
p_12=p(1):p(2); % from 0 to M
p_23=p(2):p(3); % from M to L
p_2N=p(2):N; % from M to Inf
p_3N=p(3):N; % from L to Inf

% %%%%%%%%%%% Compute the stationary law %%%%%%%%%%%

% Q1=\int_{0}^{\xi}f_{\alpha\delta,\beta}\left(x-y\right)\pi\left(y\right)dy
Q1=ntrapz(X(p_12)',repmat(Pi(p_12),1,N).*FT(:,p_12)');

% Q2=f_{\alpha\delta,\beta}\left(x\right)\cdot\int_{\xi}^{L}\left(\int_{y}^{\infty}
% f_{\alpha\psi\left(y\right),\beta}\left(z-y\right)dz\right)\pi\left(y\right)dy
inQ2=ntrapz(X(p_2N)',FD(p_2N,p_23));
Q2=ntrapz(X(p_23)',repmat(inQ2'.*Pi(p_23),1,N)).*FT(:,1)';

% Q3=f_{\alpha\delta,\beta}\left(x\right)\cdot\int_{L}^{\infty}\pi\left(y\right)dy
Q3=ntrapz(X(p_3N)',repmat(Pi(p_3N),1,N)).*FT(:,1)';

% Sum of Q
Q=Q1+Q2+Q3;

% The value of integral of stationaty law is equal 1
% ==> divide to the area under the stationaty law
Q=Q/ntrapz(X,Q);
Q=Q';