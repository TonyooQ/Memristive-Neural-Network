function [w]=binary(w0)
w= -1.*(w0<0)+1.*(w0>=0);