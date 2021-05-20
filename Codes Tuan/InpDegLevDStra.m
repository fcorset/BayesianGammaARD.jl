function [vXTAnte,vXTPost,vXIAnte]=InpDegLevDStra(a,b,L,M,delT,Dopt,Ropt,Qopt,nI,idx)
% [vXTAnte,vXTPost,vXIAnte]=InpDegLevDStra(a,b,L,M,delT,D,nI)

% a,b are parameter of gamma process
% L is the failure threshold
% M is the degradation associated with the prediction accuracy
% delT is the inspection period
% D is the waiting time before a replacement
% nI is the number of inspections
% vXTAnte and vXTPost are vectors of degradation at just before and just after an inspection or a replacement time
% vXIAnte is vector of degradation levels at just before an inspection

% This function simulate the degradation level of the system at the inspection
% and replacement times according to the (\delta,\xi,\lambda) strategy

% © 12/06/2015 Tuan Huynh - tuan.huynh@utt.fr - UTT, France 

% Pre-allocation
vXTAnte=zeros(1,nI); vXTPost=vXTAnte;
vXIAnte=nan*ones(1,nI);
vXIAnte(1)=0;

% load data to compute the waiting time
load DataForStaLawA1p3B1p3L15 % a=1/3, b=1/3, L=15

% vector of possible values of waiting time
if idx==0 % reliability-based waiting time
    nX=length(vX); vD=zeros(1,nX);
    for p=1:nX;
        vD(p)=interp1(R(:,p),u,Ropt,'nearest','extrap');
    end
elseif idx==1 % MRL-based waiting time
    vD=mrl-Qopt; vD(vD<0)=0;
end

for k=2:nI
    
    if vXTPost(k-1)<M
        vXTAnte(k)=vXTPost(k-1)+gamrnd(a*delT,1/b);
        vXIAnte(k)=vXTAnte(k);
    else
        % compute the waiting time D
        if idx==2
            D=Dopt;
        else
            D=interp1(vX,vD,vXTPost(k-1),'nearest','extrap');
        end
        vXTAnte(k)=vXTPost(k-1)+gamrnd(a*D,1/b);
    end
    
    % if the system fails
    if vXTAnte(k)>=L
        
        % Replace immediately at Tk
        vXTPost(k)=0;
    
    % if the RUL prediction is anough accurate
    elseif vXTAnte(k)>=M
        
        % decide depending on vXTPost(k-1)
        if vXTPost(k-1)>=M
            
            % Replace immediately at Tk
            vXTPost(k)=0;
        
        else
            
            % Replace D unit latter
            vXTPost(k)=vXTAnte(k);
        
        end
       
    % if the RUL prediction is inaccurate
    else        
       
        % state is still as it is
        vXTPost(k)=vXTAnte(k);        
    
    end
end

vXIAnte=vXIAnte(isfinite(vXIAnte));