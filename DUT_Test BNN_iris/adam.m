function [delta,m,v]=adam(grad,m,v,i)
beta1=0.99;beta2=0.999;eps=1e-10;
t=i-1;

m=beta1*m+(1-beta1)*grad;
v=beta2*v+(1-beta2)*grad.^2;
mhat=m./(1-beta1^t);
vhat=v./(1-beta2^t);
delta=mhat./sqrt(vhat+eps);