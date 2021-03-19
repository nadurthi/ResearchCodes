function dx=dubinctrl(t,X,xr,thr)

if X(3)>pi
    X(3)=X(3)-2*pi;
end
if X(3)<-pi
    X(3)=X(3)+2*pi;
end



% xr=[5,5,pi/2]';
xrd=[0,0,0]';

pr = xr(t);
pr=pr(:);
thr = thr(t);

th=X(3);



p = X(1:2);
p=p(:);

alpha = atan2(pr(2)-X(2),pr(1)-X(1));

R=[cos(th),sin(th),0;-sin(th),cos(th),0;0,0,1];
peW=pr-p;
peR = R*[peW;1];
the=atan2(peR(2),peR(1));


%% proportional control
kpv = 5;
kpop = 20;



V= kpv*norm(pr-p);
Om=kpop*the;
% [norm(pr-p),alpha,difftha,diffthr]
%% clip the control input to the max and min possible values
Vmin =0;
Vmax = 5;
Ommin = -5;
Ommax = 5;
V = min([max([Vmin,V]),Vmax]);
Om = min([max([Ommin,Om]),Ommax]);

[V,Om]
%%
dx=zeros(3,1);
dx(1) = V*cos(th);
dx(2) = V*sin(th);
dx(3) = Om;
