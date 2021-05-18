function [Q,X]=CompPlotNumStaLaw(a,b,L,M,delT,Dopt,Ropt,Qopt,Xinf,h,n,idx)

% This function compute and plot the stationary law of the state of the
% maintaied system under the predictive maintenance framework considering 
% improving prediction accuracy

% a,b are parameter of gamma process
% L is the failure threshold
% M is the degradation associated with the prediction accuracy
% delT is the inspection period
% Dopt, Ropt, Qopt is the optimal parameter for waiting time
% Xinf is the approximation of infty of degradation
% h is the discritization step (constant) of degradation vector X
% n is the number iterations
% idx is the index indicating the way to determine D: 
% - idx=0 => D is based on the system reliability
% - idx=1 => D is based on the system MRL
% - idx=2 => D is constant
% X is the [N, 1] vector of considered degradation levels
% Q is the [N, n] matrix of stationary law, columns corespond to stationary
% laws for each iteration.

% © 21/05/2015 Tuan Huynh - tuan.huynh@utt.fr - UTT, France 

% %%%%%%% vector of degradation levels & associated indices %%%%%%%
[X,p]=XEval([0 M L Xinf],h);

% %%%%%%% Matrix of transition probability density function %%%%%%%

N=length(X);

% when semi-renewal cycle = inspection period
FT=TransDens(X,a,b,delT,N);
FT(1:N+1:end)=0;

% when semi-renewal cycle = waiting time
if idx==2
    FD=TransDens(X,a,b,Dopt,N); 
else
    FD=TransDensD(X,a,b,N,Ropt,Qopt,idx);
end
FD(1:N+1:end)=0;

% %%%%%%% Compute the stationary law & marginal probabilities %%%%%%%
Q=ones(N,n);
Q(:,1)=StaLawFunc(X,p,FT,FD,ones(N,1),N);
for k=2:n
    Q(:,k)=StaLawFunc(X,p,FT,FD,Q(:,k-1),N);
end

%%%%%%% stationary law %%%%%%%
% figure;
% hold on,
% plot(X,Q,'-k','LineWidth',1,'MarkerFaceColor','w','MarkerSize',6)
% plot([M M],get(gca,'ylim'),'--k','LineWidth',1) % line of M
% plot([L L],get(gca,'ylim'),'--k','LineWidth',1) % line of L
% set(gca,'FontSize',12)
% ylabel('\pi(x)','FontSize',12)
% xlabel('x','FontSize',12)