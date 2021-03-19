#!/usr/bin/env python
"""
Documentation for this imm module

More details.
"""

# %%

function[x, w] = UT_sigmapoints(mu, P, M)
mu = mu(:);

% x = mu(:)';
% w = 1;
% return
% global kappa
kappa = 1;
n = length(P);
% m = 2 is 2n + 1 points

if M == 2
    if n < 4
    k = 3 - n;

    else

%         k = kappa;
k = 1;
    end

    x(1, : ) = mu';
% try
    w(1) = k / (n + k);
%     catch
%         keyboard
%     end
if min(eig(P)) <= 0
    min(eig(P))
end

    A = sqrtm((n + k) * P);
%     A = chol((n + k) * P);
   for i = 1: 1: n
       x(i+1, : ) = (mu+A(: , i))';
       x(i+n+1, : ) = (mu-A(: , i))';
       w(i + 1) = 1 / (2 * (n + k));
       w(i + n + 1) = 1 / (2 * (n + k));
   end
   w = w';
end

% m = 4 is 4n + 1 points

if M == 4
    a = normpdf(2) / normpdf(1);
    b = normpdf(3) / normpdf(1);
    x(1, : ) = mu';
    w(1) = 1 - n * (1 + a + b) / (1 + 4 * a + 9 * b);
    A = sqrtm(P);
   for i=1:1:n
       x(i+1,:)=(mu+A(:,i))';
       x(i+n+1,:)=(mu-A(:,i))';
       x(i+2*n+1,:)=(mu+2*A(:,i))';
       x(i+3*n+1,:)=(mu-2*A(:,i))';
       w(i+1)=1/(2+8*a+18*b);
       w(i+n+1)=1/(2+8*a+18*b);
       w(i+2*n+1)=a/(2+8*a+18*b);
       w(i+3*n+1)=a/(2+8*a+18*b);
   end
   w=w';
end

% m=6 is 6n+1 points

if M==6
    a=normpdf(2)/normpdf(1);
    b=normpdf(3)/normpdf(1);
    x(1,:)=mu';
    w(1)=1-n*(1+a+b)/(1+4*a+9*b);
    A=sqrtm(P);
   for i=1:1:n
       x(i+1,:)=(mu+A(:,i))';
       x(i+n+1,:)=(mu-A(:,i))';
       x(i+2*n+1,:)=(mu+2*A(:,i))';
       x(i+3*n+1,:)=(mu-2*A(:,i))';
       x(i+4*n+1,:)=(mu+3*A(:,i))';
       x(i+5*n+1,:)=(mu-3*A(:,i))';
       w(i+1)=1/(2+8*a+18*b);
       w(i+n+1)=1/(2+8*a+18*b);
       w(i+2*n+1)=a/(2+8*a+18*b);
       w(i+3*n+1)=a/(2+8*a+18*b);
       w(i+4*n+1)=b/(2+8*a+18*b);
       w(i+5*n+1)=b/(2+8*a+18*b);
   end
   w=w';
end









# %%


function d=unifrom_moments_iid(dim,N)
%% m is the vector of moments
% N is the moment(always keep it even)
% n is dimension of system

mom=@(n)((-1-1)^n+(1+1)^n)/(2^(n+1)*(n+1));

n=dim;
% combos = combntns(1:N,2);
% ind=combntns(1:length(combos),N/2);
% A=[];
% a=[1:1:N];
% for i=1:1:length(ind)
%     r=[];
%     for j=1:1:N/2
%         r=horzcat(r,combos(ind(i,j),:));
%     end
%     if sum(abs(sort(r)-a))==0
%     A=vertcat(A,r) ;
%     end
% end

% combos = combntns(0:N,n)
combos = GenerateIndex(n,(N+1)*ones(1,n));
combos(find(combos==(N+1)))=0;

x=[];
for i=1:1:length(combos)
    if sum(combos(i,:))==N
     x=vertcat(x,combos(i,:));
%      x=vertcat(x,wrev(combos(i,:)));
    end
end
size(x);
% x=vertcat(x,N/2*ones(1,n));
x=sortrows(x,-1);
nn=size(x);
% m=zeros(nn(1),1);
g=ones(nn(1),1);
h=[];
p=[];
for i=1:1:nn(1)
    for j=1:1:nn(2)
    g(i)=g(i)*mom(x(i,j));
    end
if g(i)~=0
   p=vertcat(p,g(i));
   h=vertcat(h,x(i,:));
end
end
d=[h,p];
end












# %%


function [X,w]=uniform_sigma_pts(bdd_low,bdd_up,N)
% n=length(P);
%N is the order of sigma points
%% calculate the sigma points for iid
n=length(bdd_low);
P=eye(n);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if N==4
%
%     if n==5
% r1=0.983192080250175;
% r2=0.6784669927988098;
% w1=0.04756242568370988;
% w2=0.016386741973840664;
% m=n;
%     end
%     if n==4
% r1=0.8944271909999159;
% r2=0.7071067811865476;
% w1=0.06944444444444445;
% w2=0.027777777777777776;
% m=n;
%     end
%         if n==3
% r1=0.7958224257542215;
% r2=0.7587869106393281;
% w1=0.110803324099723;
% w2=0.04189750692520776;
%    m=n;
%         end
% if n==2
% r1=0.6831300510639732;
% r2=0.8819171036881968;
% w1=0.2040816326530612;
% w2=0.04591836734693878;
% m=n;
% end
%     if n==6
% r1=0.7954844806711471;
% r2=0.7729958609899745;
% w1=0.018498622349495227;
% w2=0.0032417355491919046;
% m=4;
%
% % r1=0.8146836628687879;
% % r2=0.7587394644519632;
% % w1=0.06936400406648856;
% % w2=0.002619249237533397;
% % m=5;
%     end
%         if n==7
% r1=0.9830726897029773;
% r2=0.7467984594276957;
% w1=0.017844575631304052;
% w2=0.001116333245776404;
% m=5;
%
% % r1=0.8146836628687879;
% % r2=0.7587394644519632;
% % w1=0.06936400406648856;
% % w2=0.002619249237533397;
% % m=5;
%     end
% A=sqrtm(P);
% X=zeros(2*n+2^m*nchoosek(n,m),n);
% dr=general_conj_axis(n,m);
% for i=1:1:n
%     X(i,:)=r1*A(:,i);
%     X(i+n,:)=-r1*A(:,i);
% end
% for i=1:1:length(dr)
%     sig=0;
%     for j=1:1:n
%         sig=sig+dr(i,j)*A(:,j);
%     end
%     X(2*n+i,:)=r2*sig;
% end
% w=[w1*ones(1,2*n),w2*ones(1,length(dr))]';
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if N==6 && n==4

%    if n==4

      r1=0.986614654720553;
      r2=0.5054081058530535;
      r3=0.8495213324566961;
      w1=0.02524093860130131;
      w2=0.027777491995230833;
      w3=0.009853499570395883;
      m1=n;
      m2=3;
%    end

   A=sqrtm(P);
X=zeros(1+2*n+2^m1*nchoosek(n,m1)+2^m2*nchoosek(n,m2),n);
% X(1,:)=zeros(1,n);

for i=1:1:n
    X(i,:)=r1*A(:,i);
    X(i+n,:)=-r1*A(:,i);
end

dr1=general_conj_axis(n,m1);
for i=1:1:length(dr1)
    sig=0;
    for j=1:1:n
        sig=sig+dr1(i,j)*A(:,j);
    end
    X(2*n+i,:)=r2*sig;
end

dr2=general_conj_axis(n,m2);
for i=1:1:length(dr2)
    sig=0;
    for j=1:1:n
        sig=sig+dr2(i,j)*A(:,j);
    end
    X(2*n+length(dr1)+i,:)=r3*sig;
end
w=[w1*ones(1,2*n),w2*ones(1,length(dr1)),w3*ones(1,length(dr2)),1-w1*(2*n)-w2*(length(dr1))-w3*(length(dr2))]';

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if N==6 && n==2

%    if n==2

      r1=0.9258200997725514;
      r2=0.8749414957695937;
      r3=0.5332579116109374;
      w1=0.060493827160493854;
      w2=0.03109422247096063;
      w3=0.1181727528376811;
      m1=2;
      m2=2;
%    end

   A=sqrtm(P);
X=zeros(1+2*n+2*2^n,n);
% X(1,:)=zeros(1,n);

for i=1:1:n
    X(i,:)=r1*A(:,i);
    X(i+n,:)=-r1*A(:,i);
end

dr1=general_conj_axis(n,m1);
for i=1:1:length(dr1)
    sig=0;
    for j=1:1:n
        sig=sig+dr1(i,j)*A(:,j);
    end
    X(2*n+i,:)=r2*sig;
end

dr2=general_conj_axis(n,m2);
for i=1:1:length(dr2)
    sig=0;
    for j=1:1:n
        sig=sig+dr2(i,j)*A(:,j);
    end
    X(2*n+length(dr1)+i,:)=r3*sig;
end
w=[w1*ones(1,2*n),w2*ones(1,length(dr1)),w3*ones(1,length(dr2)),1-w1*(2*n)-w2*(length(dr1))-w3*(length(dr2))]';

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% 4D- 8th moment%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if N==8 && n==4

%     if n==4
        r1=0.9185985004650354;
        r2=0.4056290098577023;
        r3=0.7897970163891953;
        r4=0.918231359082217;
        r5=0.5610319682295122;
        r6=0.8770580193070292;
        w1=0.008062125720404502;
        w2=0.014595344864200561;
        w3=0.013047011752780219;
        w4=0.001790784328179888;
        w5=0.004699572845907843;
        w6=0.0006502632426480917;
        w0=0.02036722182060191;
        h=1.7;
%%%%%%%%%%%%%% generating directions and corresponding points***********

A=sqrtm(P);
% X=zeros(2*n+2*2^n+2*n*(n-1)+4*n*(n-1)*(n-2)/3+n*2^n+1,n);
X=[];
w=w0;

%***************  PA   ******************************
X=r1*general_conj_axis(4,1);
w=[w;w1*ones(2*n,1)];
%**************** principal diagnol**************
X=[X;r2*general_conj_axis(4,4)];
w=[w;w2*ones(2^n,1)];
%******* Plane Diagonal direction***********
X=[X;r3*general_conj_axis(4,2)];
w=[w;w3*ones(2*n*(n-1),1)];
%**********3d space diagnol*****************
X=[X;r4*general_conj_axis(4,3)];
w=[w;w4*ones(32,1)];
%***********Space multisector**********************************
   D=general_conj_axis(4,4);
    d1=repmat([h,1,1,1],length(D),1);
    d2=repmat([1,h,1,1],length(D),1);
    d3=repmat([1,1,h,1],length(D),1);
    d4=repmat([1,1,1,h],length(D),1);
X=[X;r5*d1.*D;r5*d2.*D;r5*d3.*D;r5*d4.*D];
w=[w;w5*ones(n*2^n,1)];
%%%%%%%%%%%%%%%% principal diagnol%%%%%%%%%%%%%%%%%%%%%%%
X=[X;r6*general_conj_axis(4,4)];
w=[w;w6*ones(2^n,1)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X=[zeros(1,n);X];
%     end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CUT2 for uniform 2D

if N==2 && n==2
    r=sqrt(2/3);
    w1=1/(2*n);
    X=vertcat(r*eye(n),-r*eye(n));
    w=w1*ones(2*n,1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CUT2 for uniform 3D
if N==2 && n==3
    r=1;
    w1=1/(2*n);
    X=vertcat(r*eye(n),-r*eye(n));
    w=w1*ones(2*n,1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CUT2 for uniform 4D
if N==2 && (n==4||n==5||n>=7)
    r1=1/sqrt(3);
    w1=1/(3*2^n*r1^2);
    X=r1*general_conj_axis(n,n);
    w=w1*ones(2^n,1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CUT2 for uniform 6D
if N==2 && (n==6)
    r1=1;
    w1=1/(60*r1^2);
    X=r1*general_conj_axis(n,2);
    w=w1*ones(2*n*(n-1),1);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% CUT4 for uniform 2345D
if N==4 && n>=2 && n<=5
    r1=sqrt((4+5*n)/30);
    r2=sqrt((4+5*n)/(15*n-12));
    w1=40/(4+5*n)^2;
    w2=2^(-n)*(4-5*n)^2/(4+5*n)^2;

    X1=vertcat(r1*eye(n),-r1*eye(n));
    X2=r2*general_conj_axis(n,n);
    X=[X1;X2];
    w=[w1*ones(2*n,1);w2*ones(2^n,1)];
    2*n*w1+2^n*w2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CUT4 for uniform 6D
if N==4 && (n==6)
    a1=(5/34)*(3+2*sqrt(15));
    a2=(2/85)*(75-sqrt(15));
    r1=1/sqrt(a1);
    r2=1/sqrt(a2);
    w1=a1^2/135;
    w2=a2^2/864;
    m=4;

    X1=vertcat(r1*eye(n),-r1*eye(n));
    X2=r2*general_conj_axis(n,m);
    X=[X1;X2];
    w=[w1*ones(2*n,1);w2*ones(2^m*nchoosek(n,m),1)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CUT4 for uniform 7D
if N==4 && (n==7)
    a1=(5/91)*(7+2*sqrt(35));
    a2=(1/91)*(175-2*sqrt(35));
    r1=1/sqrt(a1);
    r2=1/sqrt(a2);
    w1=a1^2/60;
    w2=a2^2/2880;
    m=5;

    X1=vertcat(r1*eye(n),-r1*eye(n));
    X2=r2*general_conj_axis(n,m);
    X=[X1;X2];
    w=[w1*ones(2*n,1);w2*ones(2^m*nchoosek(n,m),1)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CUT4 for uniform 8D
if N==4 && (n==8)
    a1=(15/88)*(2+sqrt(70));
    a2=(3/616)*(350-sqrt(70));
    r1=1/sqrt(a1);
    r2=1/sqrt(a2);
    w1=a1^2/360;
    w2=a2^2/5760;
    m=5;

    X1=vertcat(r1*eye(n),-r1*eye(n));
    X2=r2*general_conj_axis(n,m);
    X=[X1;X2];
    w=[w1*ones(2*n,1);w2*ones(2^m*nchoosek(n,m),1)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CUT6 for uniform 3D
if N==6 && (n==3)

r1=0.9281932822178025;
r2=0.5908222639251014;
r3=0.922127315308955;
r4=1;

  w1=0.0364049422903091;
  w3=0.01204816509640614;
  w2=0.06182348318095084;
  w4=0.002;



    X1=vertcat(r1*eye(n),-r1*eye(n));
    X2=r2*general_conj_axis(n,n);
    X3=r3*general_conj_axis(n,2);
    X4=r4*general_conj_axis(n,n);
    X0=zeros(1,n);
    X=[X1;X2;X3;X4;X0];
    w=[w1*ones(2*n,1);w2*ones(2^n,1);w3*ones(2^2*nchoosek(n,2),1);w4*ones(2^n,1)];
    w=[w;1-sum(w)];
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CUT6 for uniform 4D
if N==6 && (n==4)

r1 =0.9393949834393223;
r2 =0.5908515039814186;
r3 =0.9220484019251922;
r4 =1;
w1 = 0.012318810050629845;
w2 =0.03090256416527898;
w3 = 0.012054353263474751;
w4 =0.001;

m3=2;

    X1=vertcat(r1*eye(n),-r1*eye(n));
    X2=r2*general_conj_axis(n,n);
    X3=r3*general_conj_axis(n,m3);
    X4=r4*general_conj_axis(n,n);
    X0=zeros(1,n);
    X=[X1;X2;X3;X4;X0];
    w=[w1*ones(2*n,1);w2*ones(2^n,1);w3*ones(2^m3*nchoosek(n,m3),1);w4*ones(2^n,1)];
    w=[w;1-sum(w)];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% CUT6 for uniform 5D
if N==6 && (n==5)

m3 = 3;
r1 =0.923297579867076;
r2 =0.5814647380095846;
r3 =0.9276346108872094;
r4 = 1;
w1 = 0.025621778203133178;
w2 = 0.015380518482640763;
w3 =0.002906327659135788;
w4 =0.0001;



    X1=vertcat(r1*eye(n),-r1*eye(n));
    X2=r2*general_conj_axis(n,n);
    X3=r3*general_conj_axis(n,m3);
    X4=r4*general_conj_axis(n,n);
    X0=zeros(1,n);
    X=[X1;X2;X3;X4;X0];
    w=[w1*ones(2*n,1);w2*ones(2^n,1);w3*ones(2^m3*nchoosek(n,m3),1);w4*ones(2^n,1)];
    w=[w;1-sum(w)];
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CUT6 for uniform 6D
if N==6 && (n==6)

m3 = 3;
r1 =0.8894849140673744;
r2 =0.5896097525379135;
r3 =0.9370473365959192;
r4 =1;
w1 = 0.017093329255368887;
w2 = 0.007720932492425876;
w3 =0.0018236680839206938;
w4 = 0.0001;


    X1=vertcat(r1*eye(n),-r1*eye(n));
    X2=r2*general_conj_axis(n,n);
    X3=r3*general_conj_axis(n,m3);
    X4=r4*general_conj_axis(n,n);
    X0=zeros(1,n);
    X=[X1;X2;X3;X4;X0];
    w=[w1*ones(2*n,1);w2*ones(2^n,1);w3*ones(2^m3*nchoosek(n,m3),1);w4*ones(2^n,1)];
    w=[w;1-sum(w)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% CUT6 for uniform 7D
if N==6 && (n==7)

m3 = 3;
r2= 0.50676805288042;
r3 = 0.9239364353597956;
r4 =0.7488642940801419;
r1 = 1;
w1 =0.0010582010582010583;
w2 = 0.003253912147944296;
w3 =0.0014884137782096966;
w4 = 0.001;


    X1=vertcat(r1*eye(n),-r1*eye(n));
    X2=r2*general_conj_axis(n,n);
    X3=r3*general_conj_axis(n,m3);
    X4=r4*general_conj_axis(n,n);
    X0=zeros(1,n);
    X=[X1;X2;X3;X4;X0];
    w=[w1*ones(2*n,1);w2*ones(2^n,1);w3*ones(2^m3*nchoosek(n,m3),1);w4*ones(2^n,1)];
    w=[w;1-sum(w)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CUT6 for uniform 8D
if N==6 && (n==8)

r2 = 0.5649236231145955;
r3 = 0.9074852129730301;
r4 =0.8958617999051571;
r1 = 1;
w1 =0.008465608465608466;
w2 = 0.0018753991755863456;
w3 =0.0003315651657488392;
w4 = 0.00005;
m3 = 4;


    X1=vertcat(r1*eye(n),-r1*eye(n));
    X2=r2*general_conj_axis(n,n);
    X3=r3*general_conj_axis(n,m3);
    X4=r4*general_conj_axis(n,n);
    X0=zeros(1,n);
    X=[X1;X2;X3;X4;X0];
    w=[w1*ones(2*n,1);w2*ones(2^n,1);w3*ones(2^m3*nchoosek(n,m3),1);w4*ones(2^n,1)];
    w=[w;1-sum(w)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CUT6 for uniform 9D
if N==6 && (n==9)

r2 = 0.5638796285486902;
r3 = 0.9191450300180578;
r4 =0.8512874568211038;
r1 = 1;
w1 = 0.003527336860670194;
w2 = 0.000938247507796295;
w3 =0.00020474378222142052;
w4 = 0.00005;
m3 = 4;


    X1=vertcat(r1*eye(n),-r1*eye(n));
    X2=r2*general_conj_axis(n,n);
    X3=r3*general_conj_axis(n,m3);
    X4=r4*general_conj_axis(n,n);
    X0=zeros(1,n);
    X=[X1;X2;X3;X4;X0];
    w=[w1*ones(2*n,1);w2*ones(2^n,1);w3*ones(2^m3*nchoosek(n,m3),1);w4*ones(2^n,1)];
    w=[w;1-sum(w)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CUT8 for uniform 2D
if N==8 && (n==2)

r1 = 0.8094513750932916;
r2 = 0.49083117336552706;
r3 =0.48591160471710254;
r4 = 0.8565014348133138;

h3 = 2;

w1 = 0.06373310082370073;
w2 = 0.09172084193264994;
w3 =0.017024721903651428;
w4 = 0.027615398929714725;


    X1=vertcat(r1*eye(n),-r1*eye(n));
    X2=r2*general_conj_axis(n,n);
    X4=r4*general_conj_axis(n,n);
    X3=r3*scaled_conj_axis(n,h3);
    X0=zeros(1,n);
    X=[X1;X2;X3;X4;X0];
    w=[w1*ones(size(X1,1),1);w2*ones(size(X2,1),1);w3*ones(size(X3,1),1);w4*ones(size(X4,1),1)];
    w=[w;1-sum(w)];
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CUT8 for uniform 3D
if N==8 && (n==3)

m3 = 2;
h5 = 2;
r1 = 0.74662218221624;
r2 = 0.8585800181283407;
r3 =0.8611091583824281;
r4 =0.4977491909725501;
r5=0.481280599360786;
w1 = 0.03945150565289397;
w3 = 0.013126042117291037;
w2 =0.00681176194044321;
w4 = 0.03488100867160192;
w5 =0.009190123693819182;

    X1=vertcat(r1*eye(n),-r1*eye(n));
    X2=r2*general_conj_axis(n,n);
    X4=r4*general_conj_axis(n,n);
    X3=r3*general_conj_axis(n,m3);
    X5=r5*scaled_conj_axis(n,h5);
    X0=zeros(1,n);
    X=[X1;X2;X3;X4;X5;X0];
    %w=[w1*size(X1,1);w2*size(X2,1);w3*size(X3,1);w4*size(X4,1);w5*size(X5,1)];
     w=[w1*ones(size(X1,1),1);w2*ones(size(X2,1),1);w3*ones(size(X3,1),1);w4*ones(size(X4,1),1);w5*ones(size(X5,1),1)];
    w=[w;1-sum(w)];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CUT8 for uniform 5D
if N==8 && (n==5)


h6 = 1.9;
r1 = 1/sqrt(1.4);
r2 = 0.7381963341924443;
r3 =0.9151432251306368;
r4 =0.8189442986382928;
r5=0.3940256098527255;
r6=0.5000712982964949;
w1 = 0.000594291312043670;
w2 = 0.007624208411552719;
w3 =0.0013243454403687955;
w4 = 0.0007889833338539992;
w5 =0.003100615241184666;
w6 =0.0024757515655174263;

    X1=r1*general_conj_axis(n,n);
    X2=r2*general_conj_axis(n,2);
    X3=r3*general_conj_axis(n,3);
    X4=r4*general_conj_axis(n,4);
    X5=r5*general_conj_axis(n,n);
    X6=r6*scaled_conj_axis(n,h6);
    X0=zeros(1,n);
    X=[X1;X2;X3;X4;X5;X6;X0];
    %w=[w1*size(X1,1);w2*size(X2,1);w3*size(X3,1);w4*size(X4,1);w5*size(X5,1)];
     w=[w1*ones(size(X1,1),1);w2*ones(size(X2,1),1);w3*ones(size(X3,1),1);w4*ones(size(X4,1),1);w5*ones(size(X5,1),1);w6*ones(size(X6,1),1)];
    w=[w;1-sum(w)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CUT8 for uniform 6D
if N==8 && (n==6)


h5 = 1.8;
r1 = 1/sqrt(1.001);
r2 =0.786652210405957;
r3 =0.8200385903464799;
r4 =0.8770662509606209;
r5=0.5372456401205513;
r6=0.2844191310563133;

w1 = 0.000035405341031603054;
w2 = 0.003515070897384174;
w3 =0.0008463678710993542;
w4 = 0.0005480076702546519;
w5 =0.0010551069138927968;
w6 =0.0027009986671723218;

    X1=r1*general_conj_axis(n,n);
    X2=r2*general_conj_axis(n,2);
    X3=r3*general_conj_axis(n,3);
    X4=r4*general_conj_axis(n,4);
    X5=r5*scaled_conj_axis(n,h5);
    X6=r6*general_conj_axis(n,n);
    X0=zeros(1,n);
    X=[X1;X2;X3;X4;X5;X6;X0];
    %w=[w1*size(X1,1);w2*size(X2,1);w3*size(X3,1);w4*size(X4,1);w5*size(X5,1)];
     w=[w1*ones(size(X1,1),1);w2*ones(size(X2,1),1);w3*ones(size(X3,1),1);w4*ones(size(X4,1),1);w5*ones(size(X5,1),1);w6*ones(size(X6,1),1)];
    w=[w;1-sum(w)];
end














%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%+++++++++++++++++++%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% check if all points are within the boundary
if max(max(abs(X)))>1
    X=0;
    return
end
%% Transform to required position
mu=(bdd_low+bdd_up)/2;
h=-bdd_low+bdd_up;
for i=1:1:n
    X(:,i)=(h(i)/2)*X(:,i)+mu(i);
end

end



















# %%



function [X,w]=conjugate_dir_gausspts_till_8moment(mu,P)
%dimension n of the system
n=length(mu);

if n==6
   if exist('CUT86D.mat')==2
       load('CUT86D')
       X=XXcut8;
       w=WWcut8;
       A=sqrtm(P);
       for i=1:1:length(w)
       X(i,:)=A*X(i,:)'+mu;
       end
       return
   end

end

if n==2
A=sqrtm(P);
n1=A(:,1);
n2=A(:,2);
a1 = 0.23379853497231115;
a2 = 1.3867121461809555;
a4 = 0.288547926731565;
a3 =0.771286446121831;



w1 = 0.04382264267013926;
w2 = 0.1405096621714662;
w3 = 0.0009215768861610588;
w4=0.01240953967762697;


r1=1/sqrt(a1);
r2=1/sqrt(a2);
r3=1/sqrt(a3);
r4=1/sqrt(a4);

h=3;
X(1,:)=[0,0];

X(2,:)=r1*n1;
X(3,:)=-r1*n1;
X(4,:)=r1*n2;
X(5,:)=-r1*n2;

X(6,:)=r2*(n1+n2);
X(7,:)=r2*(n1-n2);
X(8,:)=-r2*(n1+n2);
X(9,:)=-r2*(n1-n2);

X(10,:)=r4*(n1+n2);
X(11,:)=r4*(n1-n2);
X(12,:)=-r4*(n1+n2);
X(13,:)=-r4*(n1-n2);

X(14,:)=r3*(n1+h*n2);
X(15,:)=r3*(n1-h*n2);
X(16,:)=r3*(-n1+h*n2);
X(17,:)=r3*(-n1-h*n2);
X(18,:)=r3*(h*n1+n2);
X(19,:)=r3*(h*n1-n2);
X(20,:)=r3*(-h*n1+n2);
X(21,:)=r3*(-h*n1-n2);

w0=1-4*w1-4*w2-4*w4-8*w3;

w=[w0,w1*ones(1,4),w2*ones(1,4),w4*ones(1,4),w3*ones(1,8)]';
for i=1:1:n
    X(:,i)=X(:,i)+mu(i);
end
return
end


if n==3

a1 = 0.1966319276379789;
a2 = 1.9427321792767849;
a3 = 0.2944016021306629;
a4 = 0.41171525390196356;
a6 = 0.5866854673078312;


w1 = 0.024631993437193266;
w2 = 0.08151009408908164;
w3 = 0.009767235524166815;
w4=  0.00577248937435553;
w6 = 0.000279472936899139;

r1=1/sqrt(a1);
r2=1/sqrt(a2);
r3=1/sqrt(a3);
r4=1/sqrt(a4);
r6=1/sqrt(a6);

h=2.74;
%%%%%%%%%%%%%% generating directions and corresponding points***********
A=sqrtm(P);
X=zeros(2*n+2*2^n+2*n*(n-1)+n*2^n+1,n);
%*******generating the CA direction***********
index=GenerateIndex(n,n*ones(1,n));
[roww,coll]=size(index);
dr=[];
for i=1:1:roww
if length(find(index(i,:)>2))==0
    dr=vertcat(dr,index(i,:));
end
end
dr
%***************  PA   ******************************
for i=1:1:n+1
    if i==1
        X(i,:)=zeros(1,n);
    else

    X(i,:)=r1*A(:,i-1);
    X(i+n,:)=-r1*A(:,i-1);

    end
end
%**************** CA - Space diagnols**************
mo=-1*ones(1,n);
for i=1:1:2^n
    rr=mo.^dr(i,:);
    sig=0;
    for j=1:1:n
        sig=sig+rr(j)*A(:,j);
    end
    X(2*n+1+i,:)=r2*sig;
    X(2*n+1+2^n+i,:)=r4*sig;
end
%*******generating the Plane Diagonal direction***********

index=GenerateIndex(n,n*ones(1,n));

dr=[];
for i=3:1:n
index(find(index==i))=0;
end

[roww,coll]=size(index);
for i=1:1:roww
if length(find(index(i,:)==0))==n-2
    dr=vertcat(dr,index(i,:));
end
end
[rowwdr,coll]=size(dr);
drr=dr(1,:);
for i=1:1:rowwdr
    [rdr,coll]=size(drr);
    dd=0;
    for j=1:1:rdr
        dd(j)=sum(abs(drr(j,:)-dr(i,:)));
    end

    if length(find(dd==0))==0
        drr=[drr;dr(i,:)];
    end
end
drr(find(drr==2))=-1;
%*********************************************
    for i=1:1:2*n*(n-1)
    sig=0;
    for j=1:1:n
        sig=sig+drr(i,j)*A(:,j);
    end

     X(2*n+1+2*2^n+i,:)=r3*sig;
    end


% *********   space multisector  ************
index=GenerateIndex(n,n*ones(1,n));
[roww,coll]=size(index);
dr=[];
for i=1:1:roww
if length(find(index(i,:)>2))==0
    dr=vertcat(dr,index(i,:));
end
end
mo=-1*ones(1,n);
p=0;
    for i=1:1:n*2^n
        p=p+1;

        rr=mo.^dr(p,:);
        if rem(i,2^n)==0
            p=0;
        end
        k=ceil(i/2^n)-1;
        pp=[ones(1,k),h,ones(1,n-k-1)];
    sig=0;
    for j=1:1:n
        sig=sig+pp(j)*rr(j)*A(:,j);
    end
     X(2*n+1+2*2^n+2*n*(n-1)+i,:)=r6*sig;
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w0=1-2*n*w1-2^n*w2-2^n*w4-2*n*(n-1)*w3-n*2^n*w6;

w=[w0,w1*ones(1,2*n),w2*ones(1,2^n),w4*ones(1,2^n),w3*ones(1,2*n*(n-1)),w6*ones(1,n*2^n)]';
for i=1:1:n
    X(:,i)=X(:,i)+mu(i);
end
return
end


if n==4

a1 = 0.20629093125597198;
a2 = 1.585407549822809;
a3 = 0.2851818320349825;
a4 = 0.5660749628608427;
a6 = 0.7889090077901053;


w1 = 0.01811008737283111;
w2 = 0.032063273384586845;
w3 = 0.006614353755080834;
w4=0.003489906522946932;
w5=0.0006510416666666666;
w6 = 0.00025218336987488566;

r1=1/sqrt(a1);
r2=1/sqrt(a2);
r3=1/sqrt(a3);
r4=1/sqrt(a4);
r5=2;
r6=1/sqrt(a6);

h=3;

end


if n==5

a1 = 0.1866956121576737;
a2 = 1.420294749459945;
a3 = 0.29836021128926843;
a4 = 0.5123685659872401;
a6 = 0.8065591548429262;


w1 = 0.010529034221546607;
w2 = 0.015144019639537572;
w3 = 0.0052828996967816825;
w4=0.0010671298950159158;
w5=0.0006510416666666666;
w6 = 0.00013776017592074394;

r1=1/sqrt(a1);
r2=1/sqrt(a2);
r3=1/sqrt(a3);
r4=1/sqrt(a4);
r5=2;
r6=1/sqrt(a6);

h=3;
end

if n==6

a1 = 0.16666666666666666;
a3 = 0.3333333333333333;
a2 = 1.251685733072443;
a4 = 0.42609204470533507;
a6 = 0.8333333333333334;

w1 = 0.006172839506172839;
w2 = 0.006913443044833937;
w3 =0.004115226337448559;
w4=0.0002183265828666806;
w5=0.0006510416666666666;
w6 = 0.00007849171328446504;

r1=1/sqrt(a1);
r2=1/sqrt(a2);
r3=1/sqrt(a3);
r4=1/sqrt(a4);
r6=1/sqrt(a6);
r5=2;

h=3;
end
% [u,s,v]=svd(P);
% A=chol(s)*u;
% A=A';

%%%%%%%%%%%%%% generating directions and corresponding points***********
A=sqrtm(P);
X=zeros(2*n+2*2^n+2*n*(n-1)+4*n*(n-1)*(n-2)/3+n*2^n+1,n);
%*******generating the CA direction***********
index=GenerateIndex(n,n*ones(1,n));
[roww,coll]=size(index);
dr=[];
for i=1:1:roww
if length(find(index(i,:)>2))==0
    dr=vertcat(dr,index(i,:));
end
end
%***************  PA   ******************************
for i=1:1:n+1
    if i==1
        X(i,:)=zeros(1,n);
    else

    X(i,:)=r1*A(:,i-1);
    X(i+n,:)=-r1*A(:,i-1);

    end
end
%**************** CA - Space diagnols**************
mo=-1*ones(1,n);
for i=1:1:2^n
    rr=mo.^dr(i,:);
    sig=0;
    for j=1:1:n
        sig=sig+rr(j)*A(:,j);
    end
    X(2*n+1+i,:)=r2*sig;
    X(2*n+1+2^n+i,:)=r4*sig;
end
%*******generating the Plane Diagonal direction***********

index=GenerateIndex(n,n*ones(1,n));

dr=[];
for i=3:1:n
index(find(index==i))=0;
end

[roww,coll]=size(index);
for i=1:1:roww
if length(find(index(i,:)==0))==n-2
    dr=vertcat(dr,index(i,:));
end
end
[rowwdr,coll]=size(dr);
drr=dr(1,:);
for i=1:1:rowwdr
    [rdr,coll]=size(drr);
    dd=0;
    for j=1:1:rdr
        dd(j)=sum(abs(drr(j,:)-dr(i,:)));
    end

    if length(find(dd==0))==0
        drr=[drr;dr(i,:)];
    end
end
drr(find(drr==2))=-1;
%*********************************************
    for i=1:1:2*n*(n-1)
    sig=0;
    for j=1:1:n
        sig=sig+drr(i,j)*A(:,j);
    end

     X(2*n+1+2*2^n+i,:)=r3*sig;
    end

%*********   3-Subspace bisectors**********
index=GenerateIndex(n,n*ones(1,n));
for i=n:-1:4
     index(find(index==i))=0;
end
index(find(index==3))=1;

[roww,coll]=size(index);
dr=[];
for i=1:1:roww
if length(find(index(i,:)==0))==n-3
    dr=vertcat(dr,index(i,:));
end
end
[rowwdr,coll]=size(dr);
drr=dr(1,:);
for i=1:1:rowwdr
    [rdr,coll]=size(drr);
    dd=0;
    for j=1:1:rdr
        dd(j)=sum(abs(drr(j,:)-dr(i,:)));
    end

    if length(find(dd==0))==0
        drr=[drr;dr(i,:)];
    end
end
drr(find(drr==2))=-1;
    for i=1:1:4*n*(n-1)*(n-2)/3
    sig=0;
    for j=1:1:n
        sig=sig+drr(i,j)*A(:,j);
    end

     X(2*n+1+2*2^n+2*n*(n-1)+i,:)=r5*sig;
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% *********   space multisector  ************
index=GenerateIndex(n,n*ones(1,n));
[roww,coll]=size(index);
dr=[];
for i=1:1:roww
if length(find(index(i,:)>2))==0
    dr=vertcat(dr,index(i,:));
end
end
mo=-1*ones(1,n);
p=0;
    for i=1:1:n*2^n
        p=p+1;

        rr=mo.^dr(p,:);
        if rem(i,2^n)==0
            p=0;
        end
        k=ceil(i/2^n)-1;
        pp=[ones(1,k),h,ones(1,n-k-1)];
    sig=0;
    for j=1:1:n
        sig=sig+pp(j)*rr(j)*A(:,j);
    end
     X(2*n+1+2*2^n+2*n*(n-1)+4*n*(n-1)*(n-2)/3+i,:)=r6*sig;
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w0=1-2*n*w1-2^n*w2-2^n*w4-2*n*(n-1)*w3-4*n*(n-1)*(n-2)/3*w5-n*2^n*w6;

w=[w0,w1*ones(1,2*n),w2*ones(1,2^n),w4*ones(1,2^n),w3*ones(1,2*n*(n-1)),w5*ones(1,4*n*(n-1)*(n-2)/3),w6*ones(1,n*2^n)]';
for i=1:1:n
    X(:,i)=X(:,i)+mu(i);
end

if isreal(X)==1
1;
else
  error('imag points ')
end

if isreal(w)==1
1;
else
  error('imag weights ')
end

if min(w)<0
  error('neg weights ')
end













# %%



function [X,w]=conjugate_dir_gausspts_till_6moment_scheme2(mu,P)
%dimension n of the system
n=length(mu);
if n==5
    r1=2.121320343559643;
    r2=1.1338934190276817;
    r3=2.9999999999999996;

w1=0.03292181069958846;
w2=0.014703360768175577;
w3=0.000685871056241427;

X1=r1*[eye(n);-eye(n)];
X2=r2*general_conj_axis(n,n);
X3=r3*general_conj_axis(n,2);
X0=zeros(1,n);
%*********************************************
X=[X0;X1;X2;X3];
w=[w1*ones(size(X1,1),1);w2*ones(size(X2,1),1);w3*ones(size(X3,1),1)];
w=[1-sum(w);w];

end
%
if n<7 && n~=5

    options=optimset('Display','off','MaxFunEvals',10000,'MaxIter',10000);
[x,f]=fsolve(@(x)moment_6th_ND_eqns(x,n),[2,3,4]',options);
r1=x(1);
r2=x(2);
r3=x(3);
w=[(7-(n-1))/r1^6;1/(2^n*r2^6);1/(2*r3^6)];
w1=w(1);
w2=w(2);
w3=w(3);
X1=r1*[eye(n);-eye(n)];
X2=r2*general_conj_axis(n,n);
X3=r3*general_conj_axis(n,2);
X0=zeros(1,n);
%*********************************************
X=[X0;X1;X2;X3];
w=[w1*ones(size(X1,1),1);w2*ones(size(X2,1),1);w3*ones(size(X3,1),1)];
w=[1-sum(w);w];


end



if n==7


r1=2.5512003554818197;
r2=0.964263097900639;
r3 =2.3255766977088315;
w1=0.01269406283896717;
w2=0.004859445930542121;
w3 =0.000395089978993786;

X1=r1*[eye(n);-eye(n)];
X2=r2*general_conj_axis(n,n);
X3=r3*general_conj_axis(n,3);
X0=zeros(1,n);
%*********************************************
X=[X0;X1;X2;X3];
w=[w1*ones(size(X1,1),1);w2*ones(size(X2,1),1);w3*ones(size(X3,1),1)];
w=[1-sum(w);w];

end


if n==8

r1=2.449489742783178;
r2=1;
r3=2.449489742783178;
w1=0.013888888888888888;
w2=0.00234375;
w3=0.0002314814814814815;


X1=r1*[eye(n);-eye(n)];
X2=r2*general_conj_axis(n,n);
X3=r3*general_conj_axis(n,3);
X0=zeros(1,n);
%*********************************************
X=[X0;X1;X2;X3];
w=[w1*ones(size(X1,1),1);w2*ones(size(X2,1),1);w3*ones(size(X3,1),1)];
w=[1-sum(w);w];

end


if n==9

w1 =0.015076391098114098;
w2 = 0.0011342717964254396;
w3 =0.0001572731368706344;
r1 = 2.3439073215294153;
r2= 1.023262223053077;
r3 =2.5342864499001747;


X1=r1*[eye(n);-eye(n)];
X2=r2*general_conj_axis(n,n);
X3=r3*general_conj_axis(n,3);
X0=zeros(1,n);
%*********************************************
X=[X0;X1;X2;X3];
w=[w1*ones(size(X1,1),1);w2*ones(size(X2,1),1);w3*ones(size(X3,1),1)];
w=[1-sum(w);w];

end


%% Tranformation of the points
A=sqrtm(P);
for i=1:1:length(w)
    X(i,:)=(A*X(i,:)'+mu)';
end

%% discard if neagative weight
if isreal(X)==1
1;
else
  error('imag points ')
end

if isreal(w)==1
1;
else
  error('imag weights ')
end

if min(w)<0
  error('neg weights ')
end
end











# %%



function [X,w]=conjugate_dir_gausspts_till_6moment(mu,P)
%dimension n of the system
n=length(mu);
% r1=1.5;
% w1=0.2;
% r2=2;
% w2=0.02;
% r3=3;
% w3=0.002;
% r=2.4142;
% the number of points in this scheme are 2n+2^n+1 points
% x0=[r1,w1,r2,w2,r3,w3,r];
r=2.4142;
if n==2
x0=[1.7,0.2,3,0.02,5,0.002];
 x=fsolve(@D2sys,x0);
f=D2sys(x)
r1=x(1);
w1=x(2);
r2=x(3);
w2=x(4);
r3=x(5);
w3=x(6);

end
if n==3
%     r=2.7320;
%     options=optimset('MaxFunEvals',5000000,'MaxIter',10000);
% x0=[3,0.3/6,4,0.3/8,5,0.3/24];%,3];
% %  x=fsolve(@D3sys,x0,options);
%  x=fmincon(@(x)(2*x(1)^8*x(2)+8*x(3)^8*x(4)+8*x(5)^8*x(6)*(2+r^8)-105)^2,x0,[],[],[],[],[0,0,0,0,0,0],[50,1,50,1,50,1],@D3sys,options);
% r1=x(1);
% w1=x(2);
% r2=x(3);
% w2=x(4);
% r3=x(5);
% w3=x(6);
% %  r=x(7);
r=2;
r1=1.6881059940166;
r2=1.545223430876;
r3=2.4158283260926;
w1=0.111809;
w2=0.0178901;
w3=0.0000750444;

end
if n==4
x0=[sqrt(3),0.2,3,0.02,5,0.002,2.4142];
x=fsolve(@D4sys,x0);
r1=x(1);
w1=x(2);
r2=x(3);
w2=x(4);
r3=x(5);
w3=x(6);
r=x(7);
end
[u,s,v]=svd(P);
A=chol(s)*u;
A=A';
A=sqrtm(P);
X=zeros(2*n+2^n+n*2^n+1,n);
index=GenerateIndex(n,n*ones(1,n));
[roww,coll]=size(index);
dr=[];
for i=1:1:roww
if length(find(index(i,:)>2))==0
    dr=vertcat(dr,index(i,:));
end
end
for i=1:1:n+1
    if i==1
        X(i,:)=zeros(1,n);
    else

    X(i,:)=r1*A(:,i-1);
    X(i+n,:)=-r1*A(:,i-1);

    end
end
mo=-1*ones(1,n);
for i=1:1:2^n
    rr=mo.^dr(i,:);
    sig=0;
    for j=1:1:n
        sig=sig+rr(j)*A(:,j);
    end
    X(2*n+1+i,:)=r2*sig;
end

% for j=0:1:n-1
% for i=1:1:2^n
%     X(2*n+1+2^n+i+j*2^n,:)=X(2*n+1+i,:).*[ones(1,j),r,ones(1,n-j-1)];
% end
% end
mo=-1*ones(1,n);
p=0;
    for i=1:1:n*2^n
        p=p+1;

        rr=mo.^dr(p,:);
        if rem(i,2^n)==0
            p=0;
        end
        k=ceil(i/2^n)-1;
        pp=[ones(1,k),r,ones(1,n-k-1)];
    sig=0;
    for j=1:1:n
        sig=sig+pp(j)*rr(j)*A(:,j);
    end
     X(2*n+1+2^n+i,:)=r3*sig;
    end
w0=1-2*n*w1-2^n*w2-n*2^n*w3
r1
w1
r2
w2
r3
w3
r
w=[w0,w1*ones(1,2*n),w2*ones(1,2^n),w3*ones(1,n*2^n)]';
for i=1:1:n
    X(:,i)=X(:,i)+mu(i);
end















# %%

function [X,w]=conjugate_dir_gausspts(mu,P)
%dimension n of the system

n=length(mu);
% the number of points in this scheme are 2n+2^n+1 points
%x=[r1,w1,r2,w2]
% [x,fval]=fsolve(@(x)[2*x(1)^2*x(2)+2^n*x(3)^2*x(4)-1,2*x(1)^4*x(2)+2^n*x(3)^4*x(4)-3,2^n*x(3)^4*x(4)-1,2^n*x(3)^6*x(4)-3],x0);
% fval
% n=2;
% x=fmincon(@(x)(2*x(1)^2+x(2)^2-3)^2,[2,4],[],[],[],[],[0.1,0.1],[20,20],@moment_4th_ND_eqns);
if n==2 || n==1
    if n==1
        w0=0.5811010092660772;
        w1=0.20498484723245053;
        w2=0.00446464813451093;
        r1=1.4861736616297834;
        r2=3.2530871022700643;

        A=sqrtm(P);
X=zeros(2*n+2^n+1,n);
X(1,:)=zeros(1,1);

X(2,:)=r1*A;
X(3,:)=-r1*A;
X(4,:)=r2*A;
X(5,:)=-r2*A;


w=[w0,w1*ones(1,2),w2*ones(1,2)]';

    else
%     w0=0.3;
%     a2=(1-sqrt(1-2*w0))/2;
%     a1=(1-a2)/2;
% r1=1/sqrt(a1);
% r2=1/sqrt(a2);
% w1=a1^2;
% w2=a2^2/2^2;
w0=0.41553535186548973;
w1=0.021681819434216532;
w2=0.12443434259941118;
r1=2.6060099476935847;
r2=1.190556300661233;
A=sqrtm(P);
X=zeros(2*n+2^n+1,n);
X(1,:)=zeros(1,2);

X(2,:)=r1*A(:,1);
X(3,:)=-r1*A(:,1);
X(4,:)=r1*A(:,2);
X(5,:)=-r1*A(:,2);

X(6,:)=r2*(A(:,1)+A(:,2));
X(7,:)=-r2*(A(:,1)+A(:,2));
X(8,:)=r2*(A(:,1)-A(:,2));
X(9,:)=-r2*(A(:,1)-A(:,2));
w=[w0,w1*ones(1,4),w2*ones(1,4)]';
    end
else
%  w0=2/(2+n);
%  a1=1/(2+n);
%  a2=n/(2+n);
% a1=2/(n+2);
% a2=(n-2)/(n+2);
% r1=1/sqrt(a1);
% r2=1/sqrt(a2);
% w1=a1^2;
% w2=a2^2/2^n;
r1=sqrt((n+2)/2);
r2=sqrt((n+2)/(n-2));
w1=4/(n+2)^2;
w2=(n-2)^2/(2^n*(n+2)^2);
% [u,s,v]=svd(P);
% A=chol(s)*u;
% A=A';
A=sqrtm(P);
X=zeros(2*n+2^n,n);

%%%%%% conjugate directions %%%%%%%%%%%%%%
% index=GenerateIndex(n,n*ones(1,n));
% [roww,coll]=size(index);
% dr=[];
% for i=1:1:roww
% if length(find(index(i,:)>2))==0
%     dr=vertcat(dr,index(i,:));
% end
% end
dr=prod_conjugate_dir(n);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%         X(1,:)=zeros(1,n);

for i=1:1:n


    X(i,:)=r1*A(:,i);
    X(i+n,:)=-r1*A(:,i);


end
% mo=-1*ones(1,n);
for i=1:1:2^n
%     r=mo.^dr(i,:);
    sig=0;
    for j=1:1:n
        sig=sig+dr(i,j)*A(:,j);
    end
    X(2*n+i,:)=r2*sig;
end
% w0=1-2*n*w1-2^n*w2;

w=[w1*ones(1,2*n),w2*ones(1,2^n)]';
end
for i=1:1:n
    X(:,i)=X(:,i)+mu(i);
end

if isreal(X)==1
1;
else
  error('imag points ')
end

if isreal(w)==1
1;
else
  error('imag weights ')
end

if min(w)<0
  error('neg weights ')
end











# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 21:20:17 2019

@author: Nagnanamus
"""

function D=third_conj_axis(n)
dr=prod_conjugate_dir(3);
C = nchoosek(1:n,3);
[rc,cc]=size(C);
[rdr,cdr]=size(dr);
D=zeros(4*n*(n-1)*(n-2)/3,n);
for i=0:rc:rdr*rc-rc
    for j=1:1:rc
   D(i+j,C(j,1))=dr(floor(i/rc)+1,1);
   D(i+j,C(j,2))=dr(floor(i/rc)+1,2);
   D(i+j,C(j,3))=dr(floor(i/rc)+1,3);
    end
end





# %%


function [T,W]=tens_prod_vec(u,v,wu,wv)
% 1 enitity is one row of any matrix u or v
% the rows of u are tensors producted with rows of v

if isempty(u)
    T=v;
    W=wv;
    return
end
if isempty(v)
    T=u;
    W=wu;
    return
end

n=size(u,1);
m=size(v,1);
T=[];
W=[];
for i=1:1:n
    T=vertcat(T,horzcat(repmat(u(i,:),m,1),v));
    W=vertcat(W,horzcat(repmat(wu(i),m,1),wv));
end
W=prod(W,2);
% W=W/sum(W);
% uc=size(u,2);
% vc=size(v,2);
% T=zeros(n*m,size(u,2)+size(v,2));
% k=1;
% for i=1:1:n
%     T(k,1:uc)=u(i,:);
%     for j=1:1:m
%         T(k,uc+1:uc+vc)=v(j,:);
%     end
%     k=k+1;
% end




# %%



function X=scaled_conj_axis(n,h)
g=general_conj_axis(n,n);
X=[];
% X=zeros(n*2^n,n);
% cnt=1;
for i=1:1:n
    p=g;
    p(:,i)=p(:,i)*h;
    X=vertcat(X,p);
% X(cnt:cnt+2^n,1:n)=h*g(:,i);
% cnt=cnt+2^n+1;
end






# %%





function dr=prod_conjugate_dir(n)

if n==1
    dr=[1;-1];
else
p=prod_conjugate_dir(n-1);
dr=zeros(2^(n),n);
dr(1:1:2^(n-1),1)=1;
dr(1:1:2^(n-1),2:1:n)=p;
dr(2^(n-1)+1:1:2^n,:)=-dr(1:1:2^(n-1),:);
end
end









# %%


function X=multi_scaled_conj_axis(n,h,m)
% m is the number of times h is repeated in a point
g=general_conj_axis(n,n);
X=[];
for i=1:1:n
    p=g;
    p(:,i)=p(:,i)*h;
    X=vertcat(X,p);
end













# %%


def sphericalpoints():
    pass
















# %%















# %%

















# %%















# %%










# %%
