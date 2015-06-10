%------------------------------------------------------------
% ---------  Reconnaissance faciale - Eigenfaces  -----------
% -----------  Jihade Tika - Morgan Lienardy  ---------------
%------------------------------------------------------------

clear all
close all
clc
% Nombre des images dans notre base de données 
nbDossiers=15;
nbImages=7;
M=40;

% std and mean choisis
um=100;
ustd=80;

%Lecture et affichage des images 
%Créer une matrice vide d'images
S=[];
figure(1);

for i=1:nbDossiers
    for j=1:nbImages
        chemin='C:\Users\Morgan\Documents\&&Boulot\ENSSAT\IMR2\Traitements_images\Eigenfaces\s';
        %chemin='C:\Users\Jihade\Documents\ENSSAT2\Traitement_images\Cariou\Projet_Eigenfaces\s';
        str=strcat(chemin,int2str(i),'\',int2str(j),'.pgm');
        img=imread(str);
        % Nombre des lignes (N1) et des colonnes (N2)
        [irow icol]=size(img);
        %creation de (N1*N2)x1 vecteurs
        temp=reshape(img',irow*icol,1);
        % S=N1*N2xM matrices après la fin de la séquence
        S=[S temp];
    end
end




% On change mean et std de toutes les images. On normalise toutes les images.
for i=1:size(S,2)
temp=double(S(:,i));
m=mean(temp);
st=std(temp);
S(:,i)=(temp-m)*ustd/st+um;
end

% Affichage des images normalisées
figure(2);
for i=1:M
str=strcat(int2str(i),'.jpg');
img=reshape(S(:,i),icol,irow);
img=img';
eval('imwrite(img,str)');
subplot(ceil(sqrt(M)),ceil(sqrt(M)),i)
imshow(img)
drawnow;
if i==3
title('Le training set normalise') %,'fontsize',18
end
end


% mean image
m=mean(S,2);  % Obtient la moyenne de chaque ligne au lieu de chaque colonne
tmimg=uint8(m); % converti en 8-bit (0<val<255)
img=reshape(tmimg,icol,irow); % prend N1*N2 vecteurs et creer N1xN2 matrices
img=img';
figure(3);
imshow(img);
title('Mean Image') %,'fontsize',18

% Change image pour manipulation
dbx=[];
for i=1:M
temp=double(S(:,i));
dbx=[dbx temp];
end

% Matrice de covariance C = A'A, L = AA'
A=dbx';
L=A*A';
% vv  : eigenvector pour L
% dd  : eigenvalue pour L=dbx'*dbx et C=dbx*dbx';
[vv dd]=eig(L);
% Tri et supp les val a 0
v=[];
d=[];
for i=1:size(vv,2)
if(dd(i,i)>1e-4)
v=[v vv(:,i)];
d=[d dd(i,i)];
end
end

% Tri croissant
[B index]=sort(d);
ind=zeros(size(index));
dtemp=zeros(size(index));
vtemp=zeros(size(v));
len=length(index);
for i=1:len
dtemp(i)=B(len+1-i);
ind(i)=len+1-index(i);
vtemp(:,ind(i))=v(:,i);
end
d=dtemp;
v=vtemp;


% Normalise eigenvectors
for i=1:size(v,2)
kk=v(:,i);
temp=sqrt(sum(kk.^2));
v(:,i)=v(:,i)./temp;
end

% Eigenvectors pour la matrice C
u=[];
for i=1:size(v,2)
temp=sqrt(d(i));
u=[u (dbx*v(:,i))./temp];
end

% Normalise pour eigenvectors
for i=1:size(u,2)
kk=u(:,i);
temp=sqrt(sum(kk.^2));
u(:,i)=u(:,i)./temp;
end


% montre les eigenfaces
figure(4);
for i=1:size(u,2)
img=reshape(u(:,i),icol,irow);
img=img';
img=histeq(img,255);
subplot(ceil(sqrt(M)),ceil(sqrt(M)),i)
imshow(img)
drawnow;
if i==3
title('Eigenfaces')%,'fontsize',18
end
end


% Trouve le poids de chaque image du training set
omega = [];
for h=1:size(dbx,2)
WW=[];
for i=1:size(u,2)
t = u(:,i)';
WeightOfImage = dot(t,dbx(:,h)');
WW = [WW; WeightOfImage];
end
omega = [omega WW];
end


% Pour une image a inserer 
% Ex : 's5\8.pgm'
InputImage = input('Please enter the name of the image and its extension \n','s');
InputImage = imread(strcat('C:\Users\Morgan\Documents\&&Boulot\ENSSAT\IMR2\Traitements_images\Eigenfaces\',InputImage));
figure(5)
subplot(1,2,1)
imshow(InputImage); colormap('gray');title('Input image')%,'fontsize',18
InImage=reshape(double(InputImage)',irow*icol,1);
temp=InImage;
me=mean(temp);
st=std(temp);
temp=(temp-me)*ustd/st+um;
NormImage = temp;
Difference = temp-m;

p = [];
aa=size(u,2);
for i = 1:aa
pare = dot(NormImage,u(:,i));
p = [p; pare];
end
ReshapedImage = m + u(:,1:aa)*p; %m est l'image moyenne, u est l'eigenvector
ReshapedImage = reshape(ReshapedImage,icol,irow);
ReshapedImage = ReshapedImage';
%Montre l'image reconstruite
subplot(1,2,2)
imagesc(ReshapedImage); colormap('gray');
title('Image reconstruite ')%,'fontsize',18

InImWeight = [];
for i=1:size(u,2)
t = u(:,i)';
WeightOfInputImage = dot(t,Difference');
InImWeight = [InImWeight; WeightOfInputImage];
end

ll = 1:M;
figure(68)
subplot(1,2,1)
stem(ll,InImWeight)
title('Poids de l image inseree')%,'fontsize',14

% Find  distance Euclidienne
e=[];
for i=1:size(omega,2)
q = omega(:,i);
DiffWeight = InImWeight-q;
mag = norm(DiffWeight);
e = [e mag];
end

kk = 1:size(e,2);
subplot(1,2,2)
stem(kk,e)
title('Distance Euclidienne de l image inseree')%,'fontsize',14

MaximumValue=max(e)  % maximum eucledian distance
MinimumValue=min(e)    % minimum eucledian distance