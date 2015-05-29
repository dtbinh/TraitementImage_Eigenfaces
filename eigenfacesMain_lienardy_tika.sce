//---------------------------------------------------------------
//---------------------------------------------------------------
//------Projet eigenfaces - Lienardy Morgan - Tika Jihade--------
//---------------------------------------------------------------
//---------------------------------------------------------------

//Nombre des images dans notre base de données 
nbDossiers=20;
nbImages=5;

//std and mean choisis
um=100;
ustd=80;

//Lecture et affichage des images 
//Créer une matrice vide d'images
S=[];
figure(1);

for i=1:nbDossiers
    for j=1:nbImages
        chemin='C:\Users\Morgan\Documents\&&Boulot\ENSSAT\IMR2\Traitements_images\Eigenfaces\s';
        //chemin='C:\Users\Jihade\Documents\ENSSAT2\Traitement_images\Cariou\Projet_Eigenfaces\s';
        str=chemin+string(i)+'/'+string(j)+'.pgm';
        img=imread(str);
        // Nombre des lignes (N1) et des colonnes (N2)
        [irow icol]=size(img);
        //creation de (N1*N2)x1 vecteurs
        temp=matrix(img',irow*icol,1);
        // S=N1*N2xM matrices après la fin de la séquence
        S=[S temp];
    end
end


//On change mean et std de toutes les images. On normalise toutes les images.
for i=1:size(S,2)
    temp=double(S(:,i));
    m=mean(temp);
    st=stdev(temp);
    S(:,i)=(temp-m)*ustd/st+um;
end

//Affichage des images normalisées
//abc=0
//for i=1:nbDossiers
//    for j=1:nbImages
//        abc=abc+1;
//        str='s'+string(i)+'/'+string(j)+'.pgm';
//        img=matrix(S(:,i),icol,irow);
//        img=img';
//        imwrite(img,str);
//        subplot(ceil(sqrt(nbDossiers*nbImages)),ceil(sqrt(nbDossiers*nbImages)),abc)
//        imshow(img)
//        drawnow;
//    end
//end


// mean image
Sd=double(S);
m=mean(Sd,'c');  // obtains the mean of each row instead of each column
tmimg=uint8(m); // converts to unsigned 8-bit integer. Values range from 0 to 255
img=matrix(tmimg,icol,irow); // takes the N1*N2x1 vector and creates a N1xN2 matrix
img=img';
figure(3);
imshow(img);

//Covariance matrix C=A'A, L=AA'
A=Sd';
L=A*A';
// vv are the eigenvector for L
// dd are the eigenvalue for both L=Sd'*Sd and C=Sd*Sd';
[vv dd]=spec(L);
// Sort and eliminate those whose eigenvalue is zero
v=[];
d=[];
for i=1:size(vv,2)
    if(dd(i,i)>1e-4)
        v=[v vv(:,i)];
        d=[d dd(i,i)];
    end
end

//sort, will return an ascending sequence
[B index]=gsort(d,'g','i');
ind=zeros(size(index,1),size(index,2));
dtemp=zeros(size(index,1),size(index,2));
vtemp=zeros(size(v,1),size(v,2));
len=length(index);
for i=1:len
    dtemp(i)=B(len+1-i);
    ind(1,i)=len+1-index(1,i);
    vtemp(:,ind(1,i))=v(:,i);
end
d=dtemp;
v=vtemp;


//Normalization of eigenvectors
for i=1:size(v,2) //access each column
    kk=v(:,i);
    temp=sqrt(sum(kk.^2));
    v(:,i)=v(:,i)./temp;
end

//Eigenvectors of C matrix
u=[];
for i=1:size(v,2)
    temp=sqrt(d(i));
    u=[u (Sd*v(:,i))./temp];
end

//Normalization of eigenvectors
for i=1:size(u,2)
    kk=u(:,i);
    temp=sqrt(sum(kk.^2));
    u(:,i)=u(:,i)./temp;
end


// show eigenfaces
figure(4);
for i=1:size(u,2)
    img=matrix(u(:,i),icol,irow);
    img=img';
    img=histeq(img,255);
    subplot(ceil(sqrt(M)),ceil(sqrt(M)),i)
    imshow(img)
    drawnow;
    if i==3
        title('Eigenfaces','fontsize',18)
    end
end


// Find the weight of each face in the training set
omega = [];
for h=1:size(Sd,2)
    WW=[];
    for i=1:size(u,2)
        t = u(:,i)';
        WeightOfImage = dot(t,Sd(:,h)');
        WW = [WW; WeightOfImage];
    end
    omega = [omega WW];
end


// Acquire new image
// Note: the input image must have a bmp or jpg extension.
// It should have the same size as the ones in your training set.
// It should be placed on your desktop
InputImage = input('Please enter the name of the image and its extension \n','s');
InputImage = imread(strcat('D:\Documents and Settings\sis26\Desktop\',InputImage));
figure(5)
subplot(1,2,1)
imshow(InputImage); colormap('gray');title('Input image','fontsize',18)
InImage=matrix(double(InputImage)',irow*icol,1);
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
matrixdImage = m + u(:,1:aa)*p; //m is the mean image, u is the eigenvector
matrixdImage = matrix(matrixdImage,icol,irow);
matrixdImage = matrixdImage';
//show the reconstructed image.
subplot(1,2,2)
imagesc(matrixdImage); colormap('gray');
title('Reconstructed image','fontsize',18)

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
title('Weight of Input Face','fontsize',14)

// Find Euclidean distance
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
title('Eucledian distance of input image','fontsize',14)

MaximumValue=max(e)  // maximum eucledian distance
MinimumValue=min(e)    // minimum eucledian distance
