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
M=nbDossiers*nbImages;

% standard dev and moyenne choisis
moy=100;
standD=80;

%Lecture et affichage des images 
%Créer une matrice vide eigValue'images
tabImages=[];
%On va parcourir les dossiers et les images par dossier
for i=1:nbDossiers
    for j=1:nbImages
        chemin='C:\Users\Morgan\Documents\&&Boulot\ENSSAT\IMR2\Traitements_images\Eigenfaces\s';
        %chemin='C:\Users\Jihade\Documents\ENSSAT2\Traitement_images\Cariou\Projet_Eigenfaces\s';
        str=strcat(chemin,int2str(i),'\',int2str(j),'.pgm');
        img=imread(str);
        % Nombre des lignes (N1) et des colonnes (N2)
        [irow icol]=size(img);
        %creation de N1*N2 vecteurs
        temp=reshape(img',irow*icol,1);
        % tabImages=N1*N2xM matrices après la fin de la séquence
        tabImages=[tabImages temp];
    end
end




% On change la moyenne et standart deviation de toutes les images. On normalise toutes les images.
for i=1:size(tabImages,2)
    temp=double(tabImages(:,i));
    standartdev=std(temp);
    moyenne=mean(temp);
    %mise a jour de nos images
    tabImages(:,i)=(temp-moyenne)*standD/standartdev+moy;
end

% Affichage des images normalisées
figure(1);
for i=1:M
    %pour l'affichage, necessite de creer des jpg
    str=strcat(int2str(i),'.jpg');
    img=reshape(tabImages(:,i),icol,irow);
    img=img';
    eval('imwrite(img,str)');
    subplot(ceil(sqrt(M)),ceil(sqrt(M)),i)
    imshow(img)
    drawnow;
    if i==3
        title('Le training set normalise')
    end
end


% mean image
moyenne=mean(tabImages,2);  % Obtient la moyenne de chaque ligne au lieu de chaque colonne
tmimg=uint8(moyenne); % converti en 8-bit (0<val<255)
img=reshape(tmimg,icol,irow); % prend N1*N2 vecteurs et creer N1xN2 matrices
%passage des lignes en colonnes
img=img';
figure(2);
imshow(img);
title('Mean Image')

% Change image pour manipulation
dbx=[];
for i=1:M
    temp=double(tabImages(:,i));
    dbx=[dbx temp];
end

% Matrice de covariance C = A'A, L = AA'
A=dbx';
L=A*A';
% eigVecL  : eigenvector pour L
% eigValL  : eigenvalue pour L=dbx'*dbx et C=dbx*dbx';
[eigVecL eigValL]=eig(L);
% Tri et supp les val a 0
eigVector=[];
eigValue=[];
for i=1:size(eigVecL,2)
    if(eigValL(i,i)>1e-4)
        eigVector=[eigVector eigVecL(:,i)];
        eigValue=[eigValue eigValL(i,i)];
    end
end

% Tri croissant
[B index]=sort(eigValue);
ind=zeros(size(index));
dtemp=zeros(size(index));
vtemp=zeros(size(eigVector));
len=length(index);
for i=1:len
    dtemp(i)=B(len+1-i);
    ind(i)=len+1-index(i);
    vtemp(:,ind(i))=eigVector(:,i);
end
eigValue=dtemp;
eigVector=vtemp;


% Normaliser les eigenvectors
for i=1:size(eigVector,2)
    NewEigValue=eigVector(:,i);
    temp=sqrt(sum(NewEigValue.^2));
    eigVector(:,i)=eigVector(:,i)./temp;
end

% Eigenvectors pour la matrice de covariance
NewEigVector=[];
for i=1:size(eigVector,2)
    temp=sqrt(eigValue(i));
    NewEigVector=[NewEigVector (dbx*eigVector(:,i))./temp];
end

% Normalise pour eigenvectors
for i=1:size(NewEigVector,2)
    NewEigValue=NewEigVector(:,i);
    temp=sqrt(sum(NewEigValue.^2));
    NewEigVector(:,i)=NewEigVector(:,i)./temp;
end


% montre les eigenfaces
figure(3);
for i=1:size(NewEigVector,2)
    img=reshape(NewEigVector(:,i),icol,irow);
    img=img';
    img=histeq(img,255);
    subplot(ceil(sqrt(M)),ceil(sqrt(M)),i)
    imshow(img)
    drawnow;
    if i==3
title('Eigenfaces')
end
end


% Trouve le poids de chaque image du training set
omega = [];
for h=1:size(dbx,2)
    matricePoids=[];
    for i=1:size(NewEigVector,2)
        t = NewEigVector(:,i)';
        poidsImage = dot(t,dbx(:,h)');
        matricePoids = [matricePoids; poidsImage];
    end
    omega = [omega matricePoids];
end


% Pour une image a inserer 
% Ex : 's5\8.pgm'
imageATest = input('Image a tester (exemple : s3\2.pgm) \n','s');
imageATest = imread(strcat('C:\Users\Morgan\Documents\&&Boulot\ENSSAT\IMR2\Traitements_images\Eigenfaces\',imageATest));
%imageATest = imread(strcat('C:\Users\Jihade\Documents\ENSSAT2\Traitement_images\Cariou\Projet_Eigenfaces\',imageATest));
figure(4)
subplot(1,2,1)
imshow(imageATest); colormap('gray');title('Input image')
InImage=reshape(double(imageATest)',irow*icol,1);
temp=InImage;
me=mean(temp);
standartdev=std(temp);
temp=(temp-me)*standD/standartdev+moy;
NormImage = temp;
Difference = temp-moyenne;

p = [];
aa=size(NewEigVector,2);
for i = 1:aa
    pare = dot(NormImage,NewEigVector(:,i));
    p = [p; pare];
end
ReshapedImage = moyenne + NewEigVector(:,1:aa)*p; %moyenne est l'image moyenne, NewEigVector est l'eigenvector
ReshapedImage = reshape(ReshapedImage,icol,irow);
ReshapedImage = ReshapedImage';
%Montre l'image reconstruite
subplot(1,2,2)
imagesc(ReshapedImage); colormap('gray');
title('Image reconstruite ')

InImWeight = [];
for i=1:size(NewEigVector,2)
    t = NewEigVector(:,i)';
    WeightOfInputImage = dot(t,Difference');
    InImWeight = [InImWeight; WeightOfInputImage];
end

ll = 1:M;
figure(5)
subplot(1,2,1)
stem(ll,InImWeight)
title('Poids de l image inseree')

% Find  distance Euclidienne
e=[];
for i=1:size(omega,2)
    q = omega(:,i);
    DiffWeight = InImWeight-q;
    mag = norm(DiffWeight);
    e = [e mag];
end

NewEigValue = 1:size(e,2);
subplot(1,2,2)
stem(NewEigValue,e)
title('Distance Euclidienne de l image inseree')

MaximumValue=max(e)  % distance eucledian maximum  
MinimumValue=min(e)    % distance eucledian minimum  