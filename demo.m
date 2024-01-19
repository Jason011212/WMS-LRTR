clear all;
close all;
clc;
addpath(genpath('./tSVD'));
addpath(genpath('./proximal_operator'));
addpath(genpath('spams-matlab'));
addpath(genpath('BM4D'));

TIR = load('abu-urban-3.mat');
DataTest1 = TIR.data;
DataTest = DataTest1./max(DataTest1(:));
mask = double(TIR.map);



  numb_dimension =5;%urban-3 

DataTest = PCA_img(DataTest, numb_dimension);

[H,W,Dim]=size(DataTest);
num=H*W;
for i=1:Dim 
    DataTest(:,:,i) = (DataTest(:,:,i)-min(min(DataTest(:,:,i)))) / (max(max(DataTest(:,:,i))-min(min(DataTest(:,:,i)))));
end 

%%%%
mask_reshape = reshape(mask, 1, num); 
anomaly_map = logical(double(mask_reshape)>0);                                                                                                                                                                                                       
normal_map = logical(double(mask_reshape)==0); 
Y=reshape(DataTest, num, Dim)';


X=DataTest;  
[n1,n2,n3]=size(X);
%% 

opts.lambda =0.06;

    opts.mu = 1e-4;
    opts.tol = 1e-8;
    opts.rho = 1.1;
    opts.max_iter = 100;
    opts.DEBUG = 0;
        p=tic;
    [ L,S,rank] = dictionary_learning_tlrr( X, opts); 

%%
 max_iter=100;  

 
lambda=0.01;

    [Z,tlrr_E,Z_rank,err_va ] = ADMM(X,L,max_iter,lambda,num,Dim,anomaly_map,normal_map);

    
     Time_TLRR(i)=toc(p);
    
%% compute AUC
E=reshape(tlrr_E, num, Dim)';
r_new=sqrt(sum(E.^2,1));
r_max = max(r_new(:));
taus = linspace(0, r_max, 5000);
PF_40=zeros(1,5000);
PD_40=zeros(1,5000); 
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r_new> tau);
  PF_40(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD_40(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
area_TLRR = sum((PF_40(1:end-1)-PF_40(2:end)).*(PD_40(2:end)+PD_40(1:end-1))/2)

f_show=reshape(r_new,[H,W]);
f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
figure('name','DTA'), imshow(f_show);

