function [O2,mu,xmu,xhat,var,vart]=BN_train(O1,gamma,beta)
eps=1e-10;
[~,N]=size(O1);
mu=mean(O1,2);

xmu=bsxfun(@minus,O1,mu);

var=mean(xmu.^2,2);%sigma2
vart=(N/(N-1))*var;

sqrtvar=sqrt(var+eps);

ivar=gamma./sqrtvar;
xhat=bsxfun(@times,xmu,1./sqrtvar);
gammax=bsxfun(@times,O1,ivar);

out=bsxfun(@plus,gammax,beta-ivar.*mu);
O2=out;
