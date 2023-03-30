% Peter Williams - MSPR - 2015 - Mini Project

%% Load data, create data set, split into training and testing
close all
clc
clear
% Switch between computers

% cl = 1;    % Asus
cl = 2;     % Thinkpad

% Load / read data
switch cl
    case 1,
    % add prtools % load data
    prppath='D:\gd\Smc\Matlab\prtools';         % Asus
    minipath='D:\GD\Smc\Multivariate Statistics\mini-project\data\';        % Asus

    case 2,
    prppath='C:\Program Files\MATLAB\R2015a\toolbox\prtools';       % Thinkpad
    minipath='C:\Users\Peter\Google Drive\Smc\Multivariate Statistics\mini-project\data\';      % Thinkpad
end
addpath(prppath);
fid=fopen([minipath 'letter-recognition.data']);
data=textscan(fid,'%s%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d','delimiter',',');
fclose(fid);
% create a cell array (labs ) containing the data labels
labs=data{1};
% create an array (fdata) and enter the features from the dataset
fdata=[];
for i=1:16;
    fdata=[fdata double(data{(i+1)})];
end
% convert labels (labs) to numbers (1-26), and put them in a colum vector (LABS)
A=strcmp(labs,'A');
B=strcmp(labs,'B');
C=strcmp(labs,'C');
D=strcmp(labs,'D');
E=strcmp(labs,'E');
F=strcmp(labs,'F');
G=strcmp(labs,'G');
H=strcmp(labs,'H');
I=strcmp(labs,'I');
J=strcmp(labs,'J');
K=strcmp(labs,'K');
L=strcmp(labs,'L');
M=strcmp(labs,'M');
N=strcmp(labs,'N');
O=strcmp(labs,'O');
P=strcmp(labs,'P');
Q=strcmp(labs,'Q');
R=strcmp(labs,'R');
S=strcmp(labs,'S');
T=strcmp(labs,'T');
U=strcmp(labs,'U');
V=strcmp(labs,'V');
W=strcmp(labs,'W');
X=strcmp(labs,'X');
Y=strcmp(labs,'Y');
Z=strcmp(labs,'Z');
LABS=double(A*1+B*2+C*3+D*4+E*5+F*6+G*7+H*8+I*9+J*10+K*11+L*12+M*13+N*14+O*15+P*16+Q*17+R*18+S*19+T*20+U*21+V*22+W*23+X*24+Y*25+Z*26);
% Split data 70% training 30% testing 
idxtr = (1:(0.7*length(fdata)));
idxtst = (((0.7*length(fdata))+1):length(fdata));
LABStr = LABS(idxtr,:);
fdatatr = fdata(idxtr,:);
LABStst = LABS(idxtst,:);
fdatatst = fdata(idxtst,:);
% Feature labels
features=['X-box    ';'Y-box    ';'width    ';'height   ';'on-pix   ';...
    'x-bar    ';'y-bar    ';'x-var    ';'y-var    ';'xy_cor   ';'x*x*y-m  ';'x*y*y-m  ';...
    'l2r-count';'xe2y-c   ';'yb2tc    ';'ye2xcor  '];
% Create training dataset with prtools
ztr = prdataset(fdatatr,LABStr,'featlab',features,...
    'lablist',{'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P'...
    ,'Q','R','S','T','U','V','W','X','Y','Z'},'name','Letter Recognition');
% And a test dataset
ztst = prdataset(fdatatst,LABStst,'featlab',features,...
    'lablist',{'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P'...
    ,'Q','R','S','T','U','V','W','X','Y','Z'},'name','Letter Recognition');

% scatter plot To have a first look at the data
figure(1);
scatterd(ztr,'legend');
title('Fig 1. - Scatter plot of all the data');

% Isolate the letter I
figure(2);
scatterd(seldat(ztr,9),'legend');
set(gca,'XLim',[0 16]);
title('Fig 3. I Alone');

%% let's try kmeans clustering - first, what's the optimum number of
% ############ Takes a long time ###################

eva=evalclusters(fdatatr,'kmeans','CalinskiHarabasz',...
    'klist',1:30);

%% recomened 2 clusters, but I want to try 26, because there are 26 classes
close all;

[kmlabs kmdata] =prkmeans(ztr,26);
figure(4);
scatterd(kmdata,'legend');
title('Fig 4. 26 Kmeans clusters');
%%  Get a visual on original labels as well

figure(5)

subplot(2,1,1)
scatterd(kmdata);
title('Kmeans');
set(gca, 'XTickLabelMode', 'Manual')
set(gca,'XTick',[],'XLabel',[]);
subplot(2,1,2)
scatterd(ztr);

title('True Labels');


%%  See if there is a better result with just 3 letters
close all;
idx = sort(vertcat(find(LABStr==12),find(LABStr==9),find(LABStr==24)));
data3=fdatatr(idx,:);
LABS3=LABStr(idx,:);


eva=evalclusters(data3,'kmeans','CalinskiHarabasz',...
    'klist',1:7);
% Gives a result of 3 clusters.... promissng

% create z3 - a prdataset containing only our 3 letters L, I and X
z3 = prdataset(data3,LABStr(idx),'featlab',['X-box    ';'Y-box    ';'width    ';'height   ';'on-pix   ';...
    'x-bar    ';'y-bar    ';'x-var    ';'y-var    ';'xy_cor   ';'x*x*y-m  ';'x*y*y-m  ';...
    'l2r-count';'xe2y-c   ';'yb2tc    ';'ye2xcor  '],...
    'lablist',{'I','L','X'},'name','Letter Recognition');

[kmlabs kmdata] =prkmeans(z3,3);

% Get a visual on this
figure(6);
title('Fig 6');
subplot(2,1,1);
scatterd(kmdata,'legend');
t = title('Clustering, Kmeans, Letters L, I and X');
set(gca, 'XTickLabelMode', 'Manual')
set(gca,'XTick',[],'XLabel',[]);
% Compare plot with original labels
subplot(2,1,2);
scatterd(z3,'legend');
t = title('Orignal labels, Letters L, I and X');

% Same plots In three dimensions
figure(7);
scatterd(kmdata,3,'legend');
t = title('Fig. 7. Clustering, Kmeans, Letters L, I and X');
t.FontSize = 10;
% original labels
figure(8);
scatterd(z3,3,'legend');
t = title('Fig. 8. Original labels, Letters L, I and X');
t.FontSize = 10;

% How succsesful was this categorisation?
trueLabels=getlabels(z3);

prconf=confmat(LABS3,kmlabs);
confmat(LABS3,kmlabs);
% Even if we knew which label is which, and reorder the confusion matrix...
% Still clearly not accurate

%% Is there a better result with linkage??
close all

l = linkage(fdatatr,'average');         % maybe not the method used in plot
% set recursion to infinite because there is a lot of data
set(0,'RecursionLimit',inf);
% Get a visual on the tree diagram
figure(9);
[Hd Td order] = dendrogram(l,'ColorThreshold',4.75);
t = title('Fig. 6. Dendrogram, threshold set to 4.75');

% scatter plot
figure(10)
IDX = cluster(l, 'cutoff',4.75);
scatter3(fdatatr(:,1),fdatatr(:,2),fdatatr(:,3), 100, IDX,'p','filled')
legend('show')
title('Fig 10, Clustering using linkage');
%%

%% Is there a better result with linkage using angles instead of distance??
close all
l = linkage(fdatatr,'average','spearman');
% set recursion to infinite because there is a lot of data
set(0,'RecursionLimit',inf);
%% Get a visual on the tree diagram
figure(99);
[Hd Td order] = dendrogram(l,'ColorThreshold',0.5);
t = title('Fig. 6. Dendrogram, threshold set to 0.5');
%%
% scatter plot
figure(100)
IDX = cluster(l, 'cutoff',0.5);
scatter3(fdatatr(:,1),fdatatr(:,2),fdatatr(:,3), 100, IDX,'p','filled')
title('Fig 100, Clustering using linkage of angles');
% Clearly a better result, but can't use it as it didn't give me any labels

%%  Correlation  Print these out
clc
letters=(1:26);
Alpha = 'A':'Z';
Rc = [];
filename=[];

% Letter by letter 


% Highest correlation values
for i = 1:length(letters);
Rc = corr(fdatatr(find(LABStr==i),:));
filename = ([sprintf('CorrelationMatix%d', letters(i)) '.txt']);
dlmwrite(filename,Rc);
% Look for high correlation +ve or -ve values
[hi hj]  =  find(abs(Rc)>.8&abs(Rc)<1);
refs = find(abs(Rc)>.8&abs(Rc)<1);
nums = Rc(refs);
result = ([nums hi hj]);
[vals index] = sort(nums,'descend');
result = result(index,:);
% disp(result);
disp(['The highest correlation values for the letter ' Alpha(i) ' are']);
for j = 1:size(result,1);
    disp([num2str(result(j)) ' from colum ' num2str(result(j+size(result,1))) ' row ' num2str(result(j+(2*size(result,1))))])
end
end
% Lowest Correlation values
for i = 1:length(letters);
Rc = corr(fdatatr(find(LABStr==i),:));
[hi hj]  =  find(abs(Rc)>0&abs(Rc)<0.008);
refs = find(abs(Rc)>0&abs(Rc)<0.008);
nums = Rc(refs);
result = ([nums hi hj]);
[vals index] = sort(nums,'descend');
result = result(index,:);
% disp(result);
disp(['The lowest correlation values for the letter ' Alpha(i) ' are']);
for k = 1:size(result,1);
    disp([num2str(result(k)) ' from colum ' num2str(result(k+size(result,1))) ' row ' num2str(result(k+(2*size(result,1))))])
end
end

%%
%% Over the entire data set
Rc = corr(fdata);
filename = ('CorrelationAllFeatures.txt');
dlmwrite(filename,Rc);
[hi hj]  =  find(abs(Rc)>0&abs(Rc)<0.008);
refs = find(abs(Rc)>0&abs(Rc)<0.008);
nums = Rc(refs);
result = ([nums hi hj]);
[vals index] = sort(nums,'descend');
result = result(index,:);
% disp(result);
disp(['The lowest correlation values for the entire data set']);
for k = 1:size(result,1);
    disp([num2str(result(k)) ' from colum ' num2str(result(k+size(result,1))) ' row ' num2str(result(k+(2*size(result,1))))])
end

[hi hj]  =  find(abs(Rc)>.8&abs(Rc)<1);
refs = find(abs(Rc)>.8&abs(Rc)<1);
nums = Rc(refs);
result = ([nums hi hj]);
[vals index] = sort(nums,'descend');
result = result(index,:);
% disp(result);
disp(['The highest correlation values for the entire data set are']);
for j = 1:size(result,1);
    disp([num2str(result(j)) ' from colum ' num2str(result(j+size(result,1))) ' row ' num2str(result(j+(2*size(result,1))))])
end
%%
%%
% Further Correlation, visual inspection
close all;
Features = (1:16);
for i = 1:15;
figure(i+10);
scatterd(ztr(:,i:i+1),'legend');
title(['Fig. ' num2str(i+10) '. Scatter plot of all the objects - Selected Features'] );
end
%% And more
close all
for i = 1:15;
figure(i+25);
scatterd(ztr(:,[1 i+1]),'legend');
title(['Fig. ' num2str(i+25) '. Scatter plot of all the objects - Selected Features'] );
end
%% And more
close all
Features = (1:16);
for i = 2:14;
figure(i+40);
scatterd(ztr(:,[2 i+1]),'legend');
title(['Fig. ' num2str(i+40) '. Scatter plot of all the objects - Selected Features'] );;
end

%% 3 D animation showing A and V only, as for features 6 and 10 they show +ve and -ve correlation respectively

figure(1);
filename = 'rotation.gif';

for n = 1:180

scatterd(seldat(ztr(:,[10 6 13]),[1 22]),3);
view(n,10);
drawnow; 

      frame = getframe(1);
      im = frame2im(frame);
      [imind,cm] = rgb2ind(im,256);
      if n == 1;
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf,'DelayTime',0.01);
      else
          imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',0.01);
      end
end


%%  PCA   
clc
close all

% Calculate the covarience martrix. and plot the cumulative eigenvalues
Cv=cov(fdatatr);
[Ve De]=eig(Cv);
[De ind]=sort(diag(De),'descend');
Ves=Ve(:,ind);
figure(51); hold on
title('Fig. 51. Eigenvalue Plot');
plot(cumsum(De)/sum(De),'Color','b','LineWidth',2,'LineStyle',':');
frac = (cumsum(De)/sum(De));
for i = 1:16;
    line([0 i],[frac(i),frac(i)],'Color','w','LineWidth',0.25,'LineStyle','-.');
    line([i i],[0,frac(i)],'Color','w','LineWidth',0.25,'LineStyle','-.');
end
line([0 8],[frac(8),frac(8)],'Color','r');
line([8 8],[0,frac(8)],'Color','r');
xlabel('Eigenvector');
ylabel('% Cumulative Eigenvalue');
set(gca,'Color',[0.75 0.75 0.85],'XTick',[1:16],'YTick',[(frac([2:2:16]))],'YTickLabel',[(floor(frac(2:2:16)*100))]);
%%
% No hard knee cut off, not that surprising as a lot of the features are
% related
disp('The eigenvector with the highest eigenvalue looks like this:')
disp(Ves(:,1));
% this contains -ve components of: horizontal and vertical box position,
% box width and height, and the number of on pixels. Also a small -ve
% amount of edge counts in vertical and horizontal directions.

% This is not to say that the other features are not important: let's look
% at the second eigenvector:
disp('The eigenvector with the second highest eigenvalue looks like this:')
disp(Ves(:,2));
% Here we see that the features which are not represented strongly by the
% first eigenvector appear in the second eigenvctor as if to add contrast
% to the projection

%% Classifiers

% ####### Discriminant

clc
prmemory(200000000);
cl=2;
switch cl;
    case 1,
method = 'nmsc';        % nearest mean classifier
w=nmsc([]);
[err Cerr pred_labs]=prcrossval(ztr,w,7);
w = nmsc(ztr);
    case 2,
method='mnc'; % 
w=nmc([]);
[err Cerr pred_labs]=prcrossval(ztr,w,7);
w = nmc(ztr);
    case 3,
method = 'ldc';        % Linear Discriminate
w=ldc([]);
[err Cerr pred_labs]=prcrossval(ztr,w,7);
w = ldc(ztr);
    case 4,
method='Quadratic Discriminate'; % quadratic discriminant
w=qdc([]);
[err Cerr pred_labs]=prcrossval(ztr,w,7);
w = qdc(ztr);
end


% SVC seemed impossible here, after one day the pc crashed

% Compute the confusion matrix
conftr=confmat(LABStr,pred_labs);
confmat(LABStr,pred_labs);
Accuracy = sum(diag(conftr))/length(ztr);
disp(['Accuracy for ' method ' algorithm is ' num2str(Accuracy*100) '%']);


d = ztst*w;
confmat(d);
conf=confmat(d);
Accuracy = sum(diag(conf))/length(ztst);
disp(['Accuracy for ' method ' algorithm, mapped onto the test set is ' num2str(Accuracy*100) '%']);
% Show the roc curves for the letter X
e = prroc(d,24);
figure(53+cl)

[e.error e.xvalues e.ylabel e.xlabel] ...
= deal(1-e.xvalues,e.error,'TP Rate','FP Rate');
plote(e)
title(['Fig.54.' num2str(cl) '. ROC curve for letter X - method ' method])

%% After PCA - Cross Validation

% Support Vector Machine and Neural Net, and K nearest neighbours took a
% long time, so I am trying PCA
prmemory(200000000);


clc
[wpca fr]=pcam(ztr,8);
ztrComp = ztr*wpca;



cl=1;
switch cl;
    case 1,
method = 'nmsc';        % nearest mean classifier
w=nmsc([]);
[err Cerr pred_labs]=prcrossval(ztrComp,w,7);
w = nmsc(ztrComp);
    case 2,
method='mnc'; % 
w=nmc([]);
[err Cerr pred_labs]=prcrossval(ztrComp,w,7);
w = nmc(ztrComp);
    case 3,
method = 'ldc';        % Linear Discriminate
w=ldc([]);
[err Cerr pred_labs]=prcrossval(ztrComp,w,7);
w = ldc(ztrComp);
    case 4,
method='Quadratic Discriminate'; % quadratic discriminant
w=qdc([]);
[err Cerr pred_labs]=prcrossval(ztrComp,w,7);
w = qdc(ztrComp);
    case 5,
% ########## The following classifiers take a long time to execute
method = 'knnc';        % nearest neighbour
w=knnc([]);
[err Cerr pred_labs]=prcrossval(ztrComp,w,7);
w = knnc(ztrComp);
    case 6
method = 'svc';        % Support Vector Machine   ########## Crashed after 1 day
w=svc([]);
[err Cerr pred_labs]=prcrossval(ztrComp,w,7);
w = svc(ztrComp);
end
% Compute the confusion matrix for the cross validation tests
conf=confmat(LABStr,pred_labs);
confmat(LABStr,pred_labs);
Accuracy = sum(diag(conf))/length(ztrComp);
disp(['Accuracy for ' method ' algorithm is ' num2str(Accuracy*100) '%']);
%%
%% test the best classifiers on the test set
clc
[wpcatst fr] = pcam(ztst,8);
ztstComp = ztst*wpcatst;
cl = 1;
switch cl;
    case 1,
method = 'nmsc';        % nearest mean classifier
w = nmsc(ztr*wpca);
    case 2,
method='mnc'; % 
w = nmc(ztr*wpca);
    case 3,
method = 'ldc';        % Linear Discriminate
w = ldc(ztr*wpca);
    case 4,
method='Quadratic Discriminate'; % quadratic discriminant
w = qdc(ztr*wpca);
    case 5,
% ########## The following classifiers take a long time to execute
prmemory(200000000);
method = 'knnc';        % nearest neighbour
w = knnc(ztr*wpca);
    case 6
method = 'svc';        % Support Vector Machine
w = svc(ztr*wpca);
end

dtst = ztstComp*w;


confmat(dtst);
conf=confmat(dtst);
Accuracy = sum(diag(conf))/length(ztst);
disp(['Accuracy for ' method ' algorithm, mapped onto the test set is ' num2str(Accuracy*100) '%']);


%% 
close all
% Binary state with L and J
idx = sort(vertcat(find(LABStr==10),find(LABStr==12)));
zb = ztr(idx,:);
[wpca frac] = pcam(zb,16); hold on;

figure(52)
plot(frac,'Color','r'); hold on;
for i = 1:length(frac);
    if mod(i,2)==0;
line([0 i],[frac(i) frac(i)],'Color',[0.9 0.9 0.9]);
line([i i],[0 frac(i)],'Color',[0.9 0.9 0.9]);
    else
line([0 i],[frac(i) frac(i)],'Color',[0.8 0.8 0.8]);
line([i i],[0 frac(i)],'Color',[0.8 0.8 0.8]);
    end
end
hold on;
title('PCA evaluation using just J and L');

%% we can use just 6 features and retain 90% of the variance

%% without pcam, just using J an L
% Do this for all classifiers
ztrb = seldat(ztr,[10 12]);
ztstb = seldat(ztst,[10 12]);
% 'Support Vector Machine';
w = svc(ztrb);
d = ztstb*w;
% 'Quadratic Discriminant';
w2 = qdc(ztrb);
d2 = ztstb*w2;
% 'K nearest neighbours';
w3 = knnc(ztrb);
d3 = ztstb*w3;
% 'Gaussian Mixture Model 2 Gaussians';
w4 = mogc(ztrb);
d4 = ztstb*w4;
% 'Gaussian Mixture Model 2 Gaussians';
w5 = mogc(ztrb);
d5 = ztstb*w5;
% 'Neural Network';
w6 = bpxnc(ztrb,100);
d6 = ztstb*w6;

disp('SVC')
confmat(d);
conf = confmat(d);
Accuracy = sum(diag(conf))/length(ztstb);
disp(['Accuracy for SVC on J and L - ' num2str(Accuracy)]);
disp('QDC');
confmat(d2);
conf = confmat(d2);
Accuracy = sum(diag(conf))/length(ztstb);
disp(['Accuracy for QDC on J and L - ' num2str(Accuracy)]);
disp('knnc');
confmat(d3);
conf = confmat(d3);
Accuracy = sum(diag(conf))/length(ztstb);
disp(['Accuracy for knnc on J and L - ' num2str(Accuracy)]);
disp('GMM');
confmat(d4);
conf = confmat(d4);
Accuracy = sum(diag(conf))/length(ztstb);
disp(['Accuracy for GMM on J and L - ' num2str(Accuracy)]);
disp('GMM - 5 Gaussians');
confmat(d5);
conf = confmat(d5);
Accuracy = sum(diag(conf))/length(ztstb);
disp(['Accuracy for GMM 5 gaussians on J and L - ' num2str(Accuracy)]);
disp('Neural Network');
confmat(d6);
conf = confmat(d6);
Accuracy = sum(diag(conf))/length(ztstb);
disp(['Accuracy for nn on J and L - ' num2str(Accuracy)]);
%%  Test them on 4 letters: J, L, M and X
close all
[wpca frac] = pcam(seldat(ztr,[10 12 18 24]),16);
figure(56);
plot(frac);
% 8 features give about 90 %

%% Final attempt to run all classifiers

%% test the best classifiers on the test set

% Filter out 2/3 of data
ztr3 = [];
ztst3 = [];
LABStr3 = [];
LABStst3 = [];

for i=1:3:length(ztr);
    ztr3=vertcat(ztr3,ztr(i,:));
    LABStr3=vertcat(LABStr3,LABStr(i,:));
end
for i=1:3:length(ztst);
    ztst3=vertcat(ztst3,ztst(i,:));
    LABStst3=vertcat(LABStst3,LABStst(i,:));
end

%%
% perform pca
dummy = ztst3;
[wpca fr] = pcam(ztst3,8);
ztst3 = dummy*wpca;
dummy = ztr3;
[wpca fr] = pcam(dummy,8);
ztr3 = ztr3*wpca;
%%
% Test the classifiers

prmemory(200000000);
cl=6;
switch cl;
    case 1,
method = 'nmsc';        % nearest mean classifier
w=nmsc([]);
[err Cerr pred_labs]=prcrossval(ztr3,w,7);
% w = nmsc(ztr*wpca);
    case 2,
method='mnc'; % 
w=nmc([]);
[err Cerr pred_labs]=prcrossval(ztr3,w,7);
w = nmc(ztr*wpca);
    case 3,
method = 'ldc';        % Linear Discriminate
w=ldc([]);
[err Cerr pred_labs]=prcrossval(ztr3,w,7);
% w = ldc(ztr*wpca);
    case 4,
method='Quadratic Discriminate'; % quadratic discriminant
w=qdc([]);
[err Cerr pred_labs]=prcrossval(ztr3,w,7);
% w = qdc(ztr*wpca);
    case 5,
% ########## The following classifiers take a long time to execute
method = 'knnc';        % nearest neighbour
w=knnc([]);
[err Cerr pred_labs]=prcrossval(ztr3,w,7);
% w = knnc(ztr*wpca);
    case 6
method = 'svc';        % Support Vector Machine   ########## Crashed after 1 day
w=svc([]);
[err Cerr pred_labs]=prcrossval(ztr3,w,7);
% w = svc(ztr*wpca);
  case 7
method = 'Gaussian Mixture Model';        % Gaussian Mixture Model
w=mogc([]);
[err Cerr pred_labs]=prcrossval(ztr3,w,7);
% w = mogc(ztr*wpca);
  case 8
method = 'neural net';        % Neural Net
w=bpxnc([]);
[err Cerr pred_labs]=prcrossval(ztr3,w,7);
% w = bpxnc(ztr*wpca);
end
% Compute the confusion matrix for the cross validation tests
conf=confmat(LABStr3,pred_labs);
confmat(LABStr3,pred_labs);
Accuracy = sum(diag(conf))/length(ztr3);
disp(['Accuracy for ' method ' algorithm is ' num2str(Accuracy*100) '%']);

%%

[w r] = featselm(ztr3,svc([]),'forward',5);
ztr3S = ztr3*w;

%%
close all
clc
clear accuracies;
prmemory(200000000);
wpca = pcam(seldat(ztr,[2 6]),8);
ztrComp = seldat(ztr,[2 6])*wpca;
wpca = pcam(seldat(ztst,[2 6]),8);
ztstComp = seldat(ztst,[2 6])*wpca;
accuracies = [];
for i = 1:2;
    w = nmc(ztr);
    d = ztst*w;
    conf = confmat(d);
    A = sum(diag(conf))/length(ztstComp);
    accuracies(1,i) = A;
    w = knnc(ztr);
    d = ztst*w;
    conf = confmat(d);
    A = sum(diag(conf))/length(ztstComp);
    accuracies(2,i) = A;
end
    
    
