clc;
close all
clear all;

myFolder = '.\Set_52';
N        = 1200;
wo       = N/2;
dpix     = (4.4/20)*10^-6;
lambda   = 660*10^-9;
NA       = 0.45;
mag      = 1;
Ncut     = 0.8;
[Kotf]   = (NA/lambda);
k        = 2*pi/lambda;
x        = (-N/2:N/2-1)*dpix;
y        = (-N/2:N/2-1)*dpix;
[X,Y]    = meshgrid(x,y);
du       = 1/(N*dpix);
umax     = 1/(2*dpix);
u        = -umax:du:umax-du;v=u;
[U,V]    = meshgrid(u,v);
freq_sq  = (U.^2 + V.^2);
R        = sqrt(X.^2+Y.^2);
scale    = 0.63;
d        = 2*10^-6;
p        = 10;

load H.mat
load apodi_2.mat

W        = windo(Kotf,N,U,V,Ncut);

if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
filePattern = fullfile(myFolder, '*.bmp');
jpegFiles = dir(filePattern);
for k = 1:length(jpegFiles)
  baseFileName = jpegFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  Im = double(imread(fullFileName));
  Inten(:,:,k)=Im(1:N,1:N);
end

lambda        = 660*10^-9;
ps            = (4.4*10^-6)/20;
NI            = 7;
zvec          = [-128 -64 -32 -16 -8 -4 -2 0 2 4 8 16 32 64 128]*10^-6;
zvec          = zvec';
zfocus1       = 0;
regparam      = 5*10^-6;
[Nx,Ny,Nz]    = size(Inten);
Nsl           = 300; 
[RePhase1,SG] = RunGaussionProcess(Inten,zfocus1,zvec,lambda,ps,Nsl,regparam);
RePhase1      = RePhase1/mean(mean(mean(Inten)));
Io            = 1; 

P_exp   =  RePhase1;
F_P_exp =  (fftshift(fft2(P_exp)));
F_P_exp =  log(abs(F_P_exp));
F_P_exp =  F_P_exp./max(max(F_P_exp));
T1       = H.*SG;
T2       = (2*H.*SG.*k)./(Io.*4.*pi.*pi.*(U.^2+V.^2));
T3       = H;
T        = T1 ; 

F_P_exp = fftshift(fft2(P_exp));
[Al,al] = Object_Para_Estimation(F_P_exp,T,Kotf,U,V,W,lambda,d);

Al      = Al ; 
K       = sqrt(U.^2+V.^2);
Objamp  = Al.*((K).^-al);
Sigamp  = Objamp.*T;

[WF_P_exp,Noise_F_P_exp] = Weiner_filter_center(F_P_exp,U,V,Al,al,T,Kotf,K,W,d,lambda);
WP_exp                   = real(ifft2(ifftshift(WF_P_exp)));

figure;
imagesc(RePhase1)
colormap(gray);
title("TIE phase image")
axis off
axis image
axis equal
caxis([-1 2.6])
figure;
imagesc(WP_exp);
colormap gray;
axis off
axis image
title("Enhanced phase image")
axis equal
caxis([-1 2.6])

function [P_uni]=uni(d,lambda,k,freq_sq,N,Ip,Im,If)
deld=(Im-Ip);
T1=fftshift((fft2((k./(d.*If)).*(deld))));
T2= (4.*pi.*pi.*freq_sq);
T2=T2./(T2.^2 + eps);
P_uni= ifft2(ifftshift(T2.*T1));
end
function [WF_AF1,Noise_AF1]=Weiner_filter_center(AF1,U,V,A,alpha,Otf,Kotf,K,W,d,lambda)
Noisespectrum=AF1.*W;
sz=size(AF1,1)/2;
K(sz+1,sz+1) =1;
NoisePower = sum(sum(abs(Noisespectrum).^2))./sum(sum(W));
Otfpower=abs(Otf).^2;
OBJamp= A.*abs(K).^(-alpha);
OBJpower= OBJamp.^2;
T1= (conj(Otf)./NoisePower);
T2= Otfpower./NoisePower;
T3= 1./OBJpower;
Filter= T1./(T2+T3);
WF_AF1= Filter.*AF1;
Noise_AF1=NoisePower;
end
function [Obj1,Obj2]=Object_Para_Estimation(NoisySkHk,Otf,Kotf,U,V,W,lambda,d)
K=sqrt(U.^2+V.^2);
CC=(K>0.3*Kotf).*(K<0.4*Kotf);
NSK=(NoisySkHk).*CC;
A = sum(sum(abs(NSK)))./sum(sum(CC));
alpha =-0.5;
OBJparam = [A alpha];
Objparaopt=@(OBJparam)Objoptfunc(OBJparam,NoisySkHk,Kotf,U,V,K,Otf,CC,W,lambda,d)
options = optimset('LargeScale','off','Algorithm','active-set','MaxFunEvals',500,'MaxIter',500,'Display','notify');
[Objpara,fval]=fminsearch(Objparaopt,OBJparam,options);
Obj1=Objpara(1);
Obj2=-Objpara(2);
end
function C= Objoptfunc(OBJparam,NoisySkHk,Kotf,U,V,K,T,CC,W,lambda,d)
sz=size(NoisySkHk,1)/2;
K(sz+1,sz+1) =1;
A= OBJparam(1,1);
alpha= OBJparam(1,2);
Objamp=A.*(abs(K).^(alpha));
Signalamp=Objamp.*(abs(T));
ZZ=W;
Noisespectrum=NoisySkHk.*ZZ;
NoisePower = sum(sum(abs(Noisespectrum).^2))./sum(sum(ZZ));
Noisefreesignalamp=((abs(NoisySkHk).^2))-NoisePower;
Noisefreesignalamp=sqrt(Noisefreesignalamp);
Error=(Noisefreesignalamp)-Signalamp;
Zloop = (K<0.7*Kotf).*((K>0.3*Kotf));
invK=1./(K);
C=sum(sum(((Error.^2.*invK).*Zloop)));
end
function W=windo(Kotf,N,U,V,noiseCF)
K=sqrt(U.^2+V.^2);
W=zeros(N,N);
W=K>noiseCF*Kotf;
W=double(W);
end


function [P_uni]=uniform(d,lambda,k,freq_sq,N,Ip,Im,If)
deld=(Im-Ip);
T1=fftshift((fft2((k./(d.*If)).*(deld))));
T2= (4.*pi.*pi.*(freq_sq) );
T2=T2./(T2.^2);
T2(isnan(T2))=0;
T2(isinf(T2))=0;
P_uni= ifft2(ifftshift(T2.*T1));
end
function [S,SS]= Sincfunc(lambda,d,U,V,Kotf,beta)

Kotf= 1*Kotf;
F_ex=1./beta;
freq=sqrt(U.^2+V.^2);


Z= [-beta^6*d  -beta^5*d -beta^4*d -beta^3*d -beta^2*d -beta*d  -d 0 d  beta*d beta^2*d beta^3*d beta^4*d  beta^5*d  beta^6*d];
Fc=[ Kotf (F_ex^(1/2))*Kotf  (F_ex^(1))*Kotf  (F_ex^(3/2))*Kotf  (F_ex^(2))*Kotf (F_ex^(5/2))*Kotf (F_ex^(3))*Kotf 0];%(F_ex^(7/2))*Kotf];
S1=(sin(pi*lambda*Z(9).*(U.^2+V.^2)))./(pi*lambda*Z(9).*(U.^2+V.^2)) ;
S1(isnan(S1))=1;
S1(isinf(S1))=1;
S2=(sin(pi*lambda*Z(10).*(U.^2+V.^2)))./(pi*lambda*Z(10).*(U.^2+V.^2));
S2(isnan(S2))=1;
S2(isinf(S2))=1;
S3=(sin(pi*lambda*Z(11).*(U.^2+V.^2)))./(pi*lambda*Z(11).*(U.^2+V.^2));
S3(isnan(S3))=1;
S3(isinf(S3))=1;
S4=(sin(pi*lambda*Z(12).*(U.^2+V.^2)))./(pi*lambda*Z(12).*(U.^2+V.^2));
S4(isnan(S4))=1;
S4(isinf(S4))=1;
S5=(sin(pi*lambda*Z(13).*(U.^2+V.^2)))./(pi*lambda*Z(13).*(U.^2+V.^2));
S5(isnan(S5))=1;
S5(isinf(S5))=1;
S6=(sin(pi*lambda*Z(14).*(U.^2+V.^2)))./(pi*lambda*Z(14).*(U.^2+V.^2));
S6(isnan(S6))=1;
S6(isinf(S6))=1;
S7=(sin(pi*lambda*Z(15).*(U.^2+V.^2)))./(pi*lambda*Z(15).*(U.^2+V.^2));
S7(isnan(S7))=1;
S7(isinf(S7))=1;

W1= freq>=Fc(2);
W2=(freq<Fc(2) & freq>=Fc(3));
W3=(freq<Fc(3) & freq>=Fc(4));
W4=(freq<Fc(4) & freq>=Fc(5));
W5=(freq<Fc(5) & freq>=Fc(6));
W6=(freq<Fc(6) & freq>=Fc(7));
W7=(freq<Fc(7) & freq>=Fc(8));


S = (S1.*W1 + S2.*W2 + S3.*W3 + S4.*W4 + S5.*W5 + S6.*W6+ S7.*W7 );

zx=d;
SS=(sin(pi*lambda*zx.*(U.^2+V.^2)))./(pi*lambda*zx.*(U.^2+V.^2));
SS(isnan(SS))=1;
SS(isinf(SS))=1;

end
function [RePhase1]= exp_TIE(N,dpix,lambda,Object,NoiseLevel,H,d,beta)

zvec=[ -(beta^6)*d -(beta^5)*d -(beta^4)*d -(beta^3)*d -(beta^2)*d -(beta^1)*d -(beta^0)*d 0 (beta^0)*d (beta^1)*d (beta^2)*d (beta^3)*d (beta^4)*d (beta^5)*d (beta^6)*d];

for k = 1:length(zvec)
   [I]=FresProp(dpix,zvec(k),lambda,N,Object);
   [I]=OTFatten(I,H);
   [I]=Addnoise(I,NoiseLevel,N);
   Inten(:,:,k)=I;
end
ps=dpix;

NI=7;
zvec=zvec';

zfocus1=0; % in the data set, the infocus plane is defaulted at zfocus1=0;
regparam=eps; %Poisson solver regularization for GP TIE


%% run Gaussian Process
[Nx,Ny,Nz]=size(Inten);
Nsl=100; %Nsl is defaulted as 50. In order to reduce the computational complexity, we divide the frequency to Nsl bins. 
%For the frequency within one bin, it shares same frequency threshold and same hyper-parameters in GP regression.

RePhase1=RunGaussionProcess(Inten,zfocus1,zvec,lambda,ps,Nsl,regparam);% recovered phase
RePhase1=RePhase1/mean(mean(mean(Inten))); % normalized phase by mean of intensity.
RePhase1=RePhase1-min(RePhase1(:)); % shift data such that the smallest element of A is 0
RePhase1=RePhase1/max(RePhase1(:)); 
%% Show result for experimental data

end
function FrequencyMesh=CalFrequency(Img,lambda, ps,dz)
[nx,ny]=size(Img);

dfx = 1/nx/ps;
dfy = 1/ny/ps;
[Kxdown,Kydown] = ndgrid(double(-nx/2:nx/2-1),double(-ny/2:ny/2-1));
k0 = find(Kxdown==0&Kydown==0);
Kxdown = Kxdown*dfx;
Kydown = Kydown*dfy;

FrequencyMesh=lambda*pi*(abs(Kxdown.^2)+abs(Kydown.^2));
FrequencyMesh=FrequencyMesh*dz/(2*pi); % Normalized for sampling step and GP regression 

%
%% show Frequency
%
% DiagF=diag(FrequencyMesh);
% figure(1);
% subplot(2,2,1);
% imagesc(FrequencyMesh);
% axis image;axis off;colormap gray
% title('FrequencyMesh');colorbar
% 
% subplot(2,2,2)
% plot([1:length(DiagF)],DiagF);
% title('Cross line');


end
function [dIdzC,S_gpC]=CombinePhase(dIdzStack, Frq_cutoff,FrequencyMesh,CoeffStack,Coeff2Stack,z,ps,lambda)
%Cutoff is the cutoff area in CosH
F = @(x) ifftshift(fft2(fftshift(x)));
Ft = @(x) ifftshift(ifft2(fftshift(x)));

[Nx,Ny,Nsl]=size(dIdzStack);

dIdzC_fft=zeros(Nx,Ny);
Maskf=zeros(Nx,Ny);


S_gpC=zeros(Nx,Ny);
du = 1/(Nx*ps);
umax = 1/(2*ps);
u = -umax:du:umax-du;v=u;
[U,V]=meshgrid(u,v);


%figure;
f0=0;
f1=0;
for k=1:Nsl
    dIdz=dIdzStack(:,:,k);
    dIdz_fft=F(dIdz);
    coeff=CoeffStack(:,k);
    S_gp=zeros(Nx,Ny);
    
    f1=Frq_cutoff(k);
    Maskf=zeros(Nx,Ny);
    Maskf(find(FrequencyMesh<=f1&FrequencyMesh>f0))=1;
    f0=f1;
    dIdzC_fft=dIdzC_fft+(dIdz_fft.*Maskf); %Only update in the update area
     
    
    for i=1:numel(coeff)
        S_gp=S_gp+ (((sin(pi*lambda*z(i).*(U.^2+V.^2))).*coeff(i))) ;
        S_gp(isnan(S_gp))=0;
        S_gp(isinf(S_gp))=0;
    end
    S_gp=S_gp./(pi*lambda.*(U.^2+V.^2));
    S_gp(isnan(S_gp))=0;
    S_gp(isinf(S_gp))=0;
    S_gpC=  S_gpC + S_gp.*Maskf;
     %{
%     Coeff=CoeffStack(:,k);
%     Coeff_f=fftshift(fft(Coeff));
%   
%   
%     Coeff2=Coeff2Stack(:,k);
%     Coeff2_f=fftshift(fft(ifftshift(Coeff2)));
    
%     Nz=length(Coeff);
%     subplot(3,2,1);
%     imagesc(Maskf);
%     axis image;axis off;colormap gray
%     title(sprintf('Mask %d',k));colorbar
%     
%     
%     subplot(3,2,3);
%     plot([1:Nz],Coeff);
%     title(sprintf('SumCoeff=%f,SumCoeff^2=%f',sum(Coeff),sum(Coeff.^2)));
%     
%     subplot(3,2,4);
%     plot([1:Nz],abs(Coeff_f));
%     title('Coeff of derivative in Fourier domain');
%     
%     subplot(3,2,5);
%     plot([1:Nz],Coeff2);
%     title(sprintf('SumCoeff=%f,SumCoeff^2=%f',sum(Coeff2),sum(Coeff2.^2)));
%     
%     subplot(3,2,6);
%     plot([1:Nz],abs(Coeff2_f));
%     title('Coeff in Fourier domain');
%     
%      
%     pause(0.05);
      
    %}
end

dIdzC=real(Ft(dIdzC_fft));
end
function [dIdz Coeff Coeff2]=GPRegression(Ividmeas, zfocus,z,Sigmaf,Sigmal,Sigman)

%Gaussion processs regression

[Nx,Ny,Nz]=size(Ividmeas);
%z=([1:Nz]');
VectorOne=ones(Nz,1);
KZ=VectorOne*z'-z*VectorOne';

K=Sigmaf*(exp(-1/2/Sigmal*KZ.^2));
L=chol(K+Sigman*eye(Nz));

z2=zfocus;

Nz2=length(z2);
VectorOne2=ones(Nz2,1);
KZ2=VectorOne*z2'-z*VectorOne2';

% for first derivative
D=Sigmaf*(exp(-1/2/Sigmal*(KZ2).^2))/(-Sigmal).*(KZ2);
%Coeff=D'/(K+Sigman*eye(Nz));
Coeff=D'/L/L';% use /L/L' to be more stable to the matrix inversion

% for regression
D2=Sigmaf*(exp(-1/2/Sigmal*(KZ2).^2));
%Coeff2=D2'/(K+Sigman*eye(Nz));
Coeff2=D2'/L/L';

%Coeff=Coeff/sum(Coeff2);
%Coeff2=Coeff2/sum(Coeff2); %Normalized,  Necessary More consideration later.

dIdz=zeros(Nx,Ny);

for k=1:Nz
    dIdz=dIdz+Ividmeas(:,:,k)*Coeff(k);
end

end
function z = poissonFFT(dzdxy,regparam)
[nx,ny]=size(dzdxy);
rows=nx; cols=ny;
%[rows,cols]=size(dzdxy);
[wx, wy] = meshgrid(([1:cols]-(fix(cols/2)+1))/(cols-mod(cols,2)), ...
    ([1:rows]-(fix(rows/2)+1))/(rows-mod(rows,2)));
wx = ifftshift(wx); wy = ifftshift(wy);
DZDXY =fft2(dzdxy);% fft2(dzdxy,rows,cols);
%divide by Poisson soln in Fourier domain
Z = (DZDXY)./(4*pi*pi*(wx.^2 + wy.^2 + regparam));
%Z = (DZDXY).*(4*pi*pi*(wx.^2 + wy.^2))./( (4*pi*pi*(wx.^2 + wy.^2)).^2+ regparam);
z = real(ifft2(Z));    %solution
z=z(1:nx,1:ny);        %get rid of zero padded area
% 
% T2= (4.*pi.*pi.*(U.^2 + V.^2));
% T2=T2./(T2.^2);
% T2(isnan(T2))=0;
% T2(isinf(T2))=0;
% P_uni= ifft2(ifftshift(T2.*T1));

end
function [RePhase,S_gpC]=RunGaussionProcess(Ividmeas,zfocus,z,lambda,ps,Nsl,regparam)
%RunGaussionProcess
%z is a column vector

[Nx,Ny,Nz]=size(Ividmeas);

% divide the frequency into Nsl bins
FrequencyMesh=CalFrequency(Ividmeas(:,:,1),lambda, ps,1);

MaxF=max(max(FrequencyMesh));
MaxF=sqrt(MaxF/(lambda/2));
Frq_cutoff=[linspace(0,1,Nsl)]*MaxF;
Frq_cutoff=Frq_cutoff.^2*lambda/2;%divide the frequency to Nsl bins. For the frequency within one bin,
%it shares same frequency threshold and same hyper-parameters in GP regression.


%calculate the hyperparameters of GP of different frequency threshold in GP
%regression
SigmafStack=zeros(Nsl,1);
SigmanStack=zeros(Nsl,1);
SigmalStack=zeros(Nsl,1);

FrqtoSc=[linspace(1.2,1.1,Nsl)];% trade off of noise and accuracy
p=Nz/(max(z)-min(z));%average data on unit space


for k=1:Nsl
%initialize Sigman and Sigmaf
Sigman=double(10^(-9));
Sigmaf=double(1);

%calculating Sigmal
f1=Frq_cutoff(k);
sc=f1*FrqtoSc(k);%FrqtoSc(k) lightly larger than 1
a=sc^2*2*(pi)^2; b=log((p*(2*pi)^(1/2))/Sigman);
%fu=@(x)a*x-0.5*log(x)-b;
fu2=@(x)a*exp(x)-0.5*x-b;
x=fzero(fu2,5);
Sigmal=double(exp(x));% Sigmal varies on sc

SigmafStack(k)=Sigmaf;
SigmanStack(k)=Sigman;
SigmalStack(k)=Sigmal;
end



dIdzStack=zeros(Nx,Ny,Nsl);% store the recover phase images for Nsl hyperparamter pairs 
CoeffStack=zeros(Nz,Nsl);
Coeff2Stack=zeros(Nz,Nsl);
%figure;
for k=1:Nsl

 Sigmal=SigmalStack(k);
 Sigman=SigmanStack(k);
 Sigmaf=SigmafStack(k);
[dIdz Coeff Coeff2]=GPRegression(Ividmeas, zfocus,z,Sigmaf,Sigmal,Sigman); %GP regression

dIdz=2*pi/(lambda)*ps^2*dIdz;
dIdzStack(:,:,k)=dIdz;
CoeffStack(:,k)=Coeff;%derivative
Coeff2Stack(:,k)=Coeff2; %smoothing, dummy paramter, which is not used afterwards in this version.

% RePhasek=poissonFFT(dIdz,regparam);
% 
% imagesc(RePhasek);
% axis image;axis off;colormap gray
% title(sprintf('Recovered Phase at %d step',k));colorbar
% pause(0.1)

%}
end


[dIdzC,S_gpC]=CombinePhase(dIdzStack, Frq_cutoff,FrequencyMesh,CoeffStack,Coeff2Stack,z,ps,lambda);%For the frequency within one bin, it shares same threshold sc and hyper-parameters in GP regression.

%% Poisson solver for pure phase 

RePhase=poissonFFT(dIdzC,regparam);

end
