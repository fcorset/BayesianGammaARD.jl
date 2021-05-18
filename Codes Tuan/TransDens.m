function y=TransDens(X,a,b,DelT,N)

% function y=TransDens(X,a,b,DelT,N)
% This function compute the matrix of transition probability density
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

y=zeros(N);
for i=1:length(X)
    y(i,1:i)=gampdf(X(i)-X(1:i),a*DelT,1/b);
end

end