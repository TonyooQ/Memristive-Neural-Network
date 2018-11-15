function w = clip(w0)
M= w0>1;N= w0<-1;
w0(M)=1;w0(N)=-1;
w=w0;