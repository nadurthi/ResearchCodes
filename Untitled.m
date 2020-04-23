clc
close all

xtrpoly = [-6,-3,-3,3,3,6, 6, 3, 3,-3,-3,-3,-6,-6,-6];
ytrpoly = [ 6, 6, 3,3,6,6,-6,-6,-3,-3,-3,-6,-6,-6, 6];

figure
plot(xtrpoly,ytrpoly,'bo-')
poly1 = polyshape(xtrpoly,ytrpoly);
lineseg=[0,0;5,8]
[in,out] = intersect(poly1,lineseg);
in
out

figure
plot(xtrpoly,ytrpoly,'bo-',in(:,1),in(:,2),'k*')

N=length(xtrpoly);
figure
plot(1:N,xtrpoly,'bo-')

figure
plot(1:N,ytrpoly,'bo-')

%%
close all
X=[];
Y=[];
p=15;
for i=0:p
    lineseg=[0,0;8*cos(i*2*pi/p),8*sin(i*2*pi/p)];
    [in,out] = intersect(poly1,lineseg);
    X=[X;in(2,1)];
    Y=[Y;in(2,2)];
end
N=length(X);
x=[1:N]';
% F=@(x)sin(x);
y=X;
rng default
gprMdl2 = fitrgp(x,y,'KernelFunction','squaredexponential',...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus'));

gprMdl2y = fitrgp(x,Y,'KernelFunction','squaredexponential',...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus'));

ypred2 = resubPredict(gprMdl2);
ypred2y = resubPredict(gprMdl2y);

xpred = linspace(1,N,100)';
[ypred,ystd] = predict(gprMdl2,xpred);
[ypredy,ystdy] = predict(gprMdl2y,xpred);

figure
plot(x,y,'r*',xpred,ypred,'b',xpred,ypred+ystd,'k',xpred,ypred-ystd,'k')


ssx=ypred;
ssx(ypred>=0)=ssx(ypred>=0)+ystd(ypred>=0);
ssx(ypred<0)=ssx(ypred<0)-ystd(ypred<0);
ssy=ypredy;
ssy(ypredy>=0)=ssy(ypredy>=0)+ystdy(ypredy>=0);
ssy(ypredy<0)=ssy(ypredy<0)-ystdy(ypredy<0);

figure
plot(xtrpoly,ytrpoly,'bo-',ypred,ypredy,'k',ssx,ssy,'r')


