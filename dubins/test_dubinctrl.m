clear
clc
close all
%%
x0=[0,0,pi/2]';
T=linspace(0,15,200);
options = odeset('RelTol',1e-6,'AbsTol',1e-6);
[t,x]=ode45(@dubinmodel,T,x0,options);

figure
plot(x(:,1),x(:,2))

%%
a=2;
xr=@(t)[sqrt(t)*sin(a*t);t];
thr=@(t)atan2(a*cos(a*t),1);

xrd=[0,0,pi/2];
x0=[4,1,pi/2]';
T=linspace(0,15,200);
options = odeset('RelTol',1e-6,'AbsTol',1e-6);
[t,x]=ode45(@dubinctrl,T,x0,options,xr,thr);

Xr = zeros(length(T),2);
for i=1:1:length(T)
    Xr(i,:)=xr(T(i));
end
%% plotting
d=2;
figure()
for i=1:1:length(t)
plot(x(1:i,1),x(1:i,2))
hold on
arr=[x(i,1),x(i,2);
    x(i,1)+d*cos(x(i,3)),x(i,2)+d*sin(x(i,3))];
plot(arr(:,1),arr(:,2),'r')
plot(x(i,1),x(i,2),'ro','MarkerSize',10)

plot(x(1,1),x(1,2),'ro','MarkerSize',10)
arr=[x0(1),x0(2);
    x0(1)+d*cos(x0(3)),x0(2)+d*sin(x0(3))];
plot(arr(:,1),arr(:,2),'r')

plot(Xr(1:i,1),Xr(1:i,2),'b','MarkerSize',10)
% arr=[xr(1),xr(2);
%     xr(1)+d*cos(xr(3)),xr(2)+d*sin(xr(3))];
% plot(arr(:,1),arr(:,2),'b')
axis([-5,5,-5,20])
pause(0.2)
hold off
end