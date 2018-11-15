function [O2]=BN_test(O1,gamma,beta,mu,vart)
eps=1e-10;

sqrtvar=sqrt(vart+eps);

ivar=gamma./sqrtvar;

gammax=bsxfun(@times,O1,ivar);

out=bsxfun(@plus,gammax,beta-ivar.*mu);
O2=out;