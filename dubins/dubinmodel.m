function dx=dubinmodel(t,X)

if X(3)>pi
    X(3)=X(3)-2*pi;
end
if X(3)<-pi
    X(3)=X(3)+2*pi;
end




th=X(3);

V=2;
Om=0.2;
%%
dx=zeros(3,1);
dx(1) = V*cos(th);
dx(2) = V*sin(th);
dx(3) = Om;