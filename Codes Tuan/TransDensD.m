function y=TransDensD(X,a,b,N,Ropt,Qopt,idx)

% function y=TransDensD(X,a,b,N,idx)
% This function compute the matrix of transition probability density when
% the waiting time interval D is function of degradation level

% a, b are parameters of degradation process
% N is the length of X
% Ropt, Qopt are reliability and safety thresholds to compute D
% idx is the index indicating the way to determine D: 
% * idx=0 => D is based on the system reliability
% * idx=1 => D is based on the system MRL
% y = NxN matrix having the following form 
%
%            ------------------------------------------------------------>
%     | [ | f(0)||     0    ||  .  ||     0    ||  .  ||  .  ||  .  ||  .  |]     
%     | [ |  .  ||    f(0)  ||  .  ||     0    ||  .  ||  .  ||  .  ||  .  |]
%     | [ |  .  ||     .    ||  .  ||     0    ||  .  ||  .  ||  .  ||  .  |]
% y = | [ |  .  ||     .    ||  .  ||    f(0)  ||  .  ||  .  ||  .  ||  .  |]
%     | [ |  .  ||f(x1i-y1k)||  .  ||f(x1i-y1i)||  .  ||  .  ||  .  ||  .  |]
%     | [ |f(x1)||     .    ||  .  ||     .    ||  .  ||  .  || f(0)||  .  |]
%     v [ |  .  ||     .    ||  .  ||f(x1j-y1i)||  .  ||  .  ||  .  || f(0)|]

% horizontal axis = y, vertical axis = x
% if x < y, then 0, else different than 0

% © 21/05/2015 Tuan Huynh - tuan.huynh@utt.fr - UTT, France 

% load data to compute the waiting time
load DataForStaLawA1p3B1p3L15 % a=1/3, b=1/3, L=15

% vector of possible values of waiting time
if idx==0 % reliability-based waiting time
    nX=length(vX); vD=zeros(1,nX);
    for p=1:nX;
        vD(p)=interp1(R(:,p),u,Ropt,'nearest','extrap');
    end
else % MRL-based waiting time
    vD=mrl-Qopt; vD(vD<0)=0;
end
    
y=zeros(N);
for i=1:length(X)
    % compute the waiting time D
    D=interp1(vX,vD,X(1:i),'nearest','extrap');
    % compute the transition probability density
    y(i,1:i)=gampdf(X(i)-X(1:i),a*D,1/b);
end

end