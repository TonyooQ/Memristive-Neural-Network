function [dO1,dgamma,dbeta]=BBN(dO2,xmu,var,gamma,xhat)
eps=1e-10;

[~,N]=size(dO2);

dbeta=mean(dO2,2);

dxhat = bsxfun(@times, dO2, gamma);
dgamma = mean(dO2.*xhat,2);%M行1列
ivar=1./sqrt(var+eps);
dvar = -0.5 * sum(dxhat .* xmu,2) .* (ivar.^3);%M行1列  
dmu = bsxfun(@times, dxhat, ivar);%M行N列
dmu = -1 * sum(dmu,2) -2 .* dvar .* mean(xmu,2);%M行1列  
di1 = bsxfun(@times,dxhat,ivar);%M行N列  
di2 = 2/N * bsxfun(@times,dvar,xmu);%M行N列   
dO1 = di1 + di2 + 1/N * repmat(dmu,1,N); 
