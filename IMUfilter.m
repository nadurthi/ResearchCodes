close all
muu=0.1;
N=50;
beta=ones(N,1)/N;
X=zeros(size(Aaccel,1));
for i=N:size(Aaccel,1)
% Aaccel(i-9:i,3)\Aaccel(i-9:i,1)
    Z=Aaccel(i-(N-1):i,3);
    e=Aaccel(i,1)-dot(beta,Z);
    beta=beta + muu*e*Z;
    beta=abs(beta);
    beta=beta/sum(beta);
    X(i)=e;
end

plot(T,Aaccel(:,1),'r',T,X(:,1))
legend('true','estimated')

%%


N=20;
X=zeros(size(Aaccel,1),1);
for i=N:size(Aaccel,1)-N
    i
    x = T(i-(N-1):i+N);
    y=Aaccel(i-(N-1):i+N,1);
    f = fit( x', y, 'fourier5');
    X(i)=f(T(i));
end
plot(T,Aaccel(:,1),'r',T,X(:,1),'b')
legend('true','estimated')

