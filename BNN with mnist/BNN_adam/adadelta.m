function [delta,Ed0,Eg0]=adadelta(grad,r,Eg0,Ed0)
eps=1e-10;

Eg=r*Eg0+(1-r).*grad.^2;
RMSg=sqrt(Eg+eps);
RMSd0=sqrt(Ed0+eps);
delta=(RMSg./RMSd0).*grad;
Ed=r*Ed0+(1-r)*delta.^2;
Eg0=Eg;Ed0=Ed;