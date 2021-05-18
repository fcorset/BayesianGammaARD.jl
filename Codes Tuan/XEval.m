function [X,p]=XEval(x,h)
% function [X,p]=XEval(x,h)
% This function create a degradation vector for evaluate the stationary law
% x is the threshold vector
% h is the discritization step (constant)
% X is the degradation vector containing the thresholds in x
% p is the position of the thresholds x in degradation vector X

% © 02/12/2012 Tuan Huynh - tuan.huynh@utt.fr - UTT, France 

n=length(x)-1;
X=0:h:x(1);
p(1)=size(X,2);
% x(1) non divisible par h ... on rajoute un "bout de pas"
if (x(1)>h)&&(X(p(1))~=x(1))
    p(1)=p(1)+1;
    X(p(1))=x(1);
end
for i=2:n+1   % car on va jusqu'a L
    if x(i) >= (x(i-1)+h) 
        X=[X x(i-1)+h:h:x(i)];
        p(i)=size(X,2);
        % x(i)-x(i-1) non divisible par h ... on rajoute un "bout de pas"
        if X(p(i))~=x(i)            
            p(i)=p(i)+1;
            X(p(i))=x(i);
        end
    elseif x(i)>x(i-1)
        p(i)=p(i-1)+1;
        X(p(i))=x(i);
    else
        p(i)=p(i-1);
    end
end