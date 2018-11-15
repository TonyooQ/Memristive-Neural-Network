function [dO1,dgamma,dbeta]=BBN(dO2,xmu,var,gamma,xhat)
eps=1e-10;

[~,N]=size(dO2);

dbeta=mean(dO2,2);

dxhat = bsxfun(@times, dO2, gamma);
dgamma = mean(dO2.*xhat,2);%M��1��
ivar=1./sqrt(var+eps);
dvar = -0.5 * sum(dxhat .* xmu,2) .* (ivar.^3);%M��1��  
dmu = bsxfun(@times, dxhat, ivar);%M��N��
dmu = -1 * sum(dmu,2) -2 .* dvar .* mean(xmu,2);%M��1��  
di1 = bsxfun(@times,dxhat,ivar);%M��N��  
di2 = 2/N * bsxfun(@times,dvar,xmu);%M��N��   
dO1 = di1 + di2 + 1/N * repmat(dmu,1,N); 
