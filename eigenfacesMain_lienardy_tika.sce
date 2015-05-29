//---------------------------------------------------------------
//---------------------------------------------------------------
//------Projet eigenfaces - Lienardy Morgan - Tika Jihade--------
//---------------------------------------------------------------
//---------------------------------------------------------------

//Nombre des images dans notre base de données 
M=40

//std and mean choisis
um=100;
ustd=80;

//Lecture et affichage des images 
//Créer une matrice vide d'images
S[];
figure(1);
for i=1:M
    //str=strcat(int2str(i),'.bmp');    % concatenates two strings that form the name of the image
    eval('img=imread(str);');
    subplot(ceil(sqrt(M)),ceil(sqrt(M)),i)
    imshow(img)
    if i==3
        title('Training set','fontsize',18)
    end
    drawnow;
    [irow icol]=size(img);    % get the number of rows (N1) and columns (N2)
    temp=reshape(img',irow*icol,1);    % creates a (N1*N2)x1 vector
    S=[S temp];    % S is a N1*N2xM matrix after finishing the sequence
end


% Here we change the mean and std of all images. We normalize all images.
% This is done to reduce the error due to lighting conditions and background.
for i=1:size(S,2)
temp=double(S(:,i));
m=mean(temp);
st=std(temp);
S(:,i)=(temp-m)*ustd/st+um;
end

% show normalized images
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
title('Normalized Training Set','fontsize',18)
end
end


% mean image
m=mean(S,2);  % obtains the mean of each row instead of each column
tmimg=uint8(m); % converts to unsigned 8-bit integer. Values range from 0 to 255
img=reshape(tmimg,icol,irow); % takes the N1*N2x1 vector and creates a N1xN2 matrix
img=img';
figure(3);
imshow(img);
title('Mean Image','fontsize',18)

% Change image for manipulation
dbx=[];    % A matrix
for i=1:M
temp=double(S(:,i));
dbx=[dbx temp];
end

%Covariance matrix C=A'A, L=AA'
A=dbx';
L=A*A';
% vv are the eigenvector for L
% dd are the eigenvalue for both L=dbx'*dbx and C=dbx*dbx';
[vv dd]=eig(L);
% Sort and eliminate those whose eigenvalue is zero
v=[];
d=[];
for i=1:size(vv,2)
if(dd(i,i)>1e-4)
v=[v vv(:,i)];
d=[d dd(i,i)];
end
end

%sort, will return an ascending sequence
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


%Normalization of eigenvectors
for i=1:size(v,2) %access each column
kk=v(:,i);
temp=sqrt(sum(kk.^2));
v(:,i)=v(:,i)./temp;
end

%Eigenvectors of C matrix
u=[];
for i=1:size(v,2)
temp=sqrt(d(i));
u=[u (dbx*v(:,i))./temp];
end

%Normalization of eigenvectors
for i=1:size(u,2)
kk=u(:,i);
temp=sqrt(sum(kk.^2));
u(:,i)=u(:,i)./temp;
end


% show eigenfaces
figure(4);
for i=1:size(u,2)
img=reshape(u(:,i),icol,irow);
img=img';
img=histeq(img,255);
subplot(ceil(sqrt(M)),ceil(sqrt(M)),i)
imshow(img)
drawnow;
if i==3
title('Eigenfaces','fontsize',18)
end
end


% Find the weight of each face in the training set
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


% Acquire new image
% Note: the input image must have a bmp or jpg extension.
% It should have the same size as the ones in your training set.
% It should be placed on your desktop
InputImage = input('Please enter the name of the image and its extension \n','s');
InputImage = imread(strcat('D:\Documents and Settings\sis26\Desktop\',InputImage));
figure(5)
subplot(1,2,1)
imshow(InputImage); colormap('gray');title('Input image','fontsize',18)
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
ReshapedImage = m + u(:,1:aa)*p; %m is the mean image, u is the eigenvector
ReshapedImage = reshape(ReshapedImage,icol,irow);
ReshapedImage = ReshapedImage';
%show the reconstructed image.
subplot(1,2,2)
imagesc(ReshapedImage); colormap('gray');
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

% Find Euclidean distance
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

MaximumValue=max(e)  % maximum eucledian distance
MinimumValue=min(e)    % minimum eucledian distance
