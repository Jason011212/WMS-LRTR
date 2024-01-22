function [ Z,E,Z_rank,err ] = ADMM(X,A,max_iter,lambda,num,Dim,anomaly_map,normal_map)
addpath PROPACK;
%%  X=A*W+E    W=Z
[n1,n2,n3]=size(X);
[~,n4,~]=size(A); 

%% Z=J=Y1 n4 n2 n3
Z=zeros(n4,n2,n3);   
J=Z;  %W
Y2=Z;   %Q2


%% E=Y2 n1 n2 n3
E = zeros(n1,n2,n3);
Y1=E;  %Q1
S=E;  %Y
Y3=E; %Q3
beta = 0.1;
miu=0.01;
max_beta = 1e+8;
tol = 1e-8;
rho = 1.1;
iter = 0;

Ain = t_inverse(A);
AT = tran(A); 
while iter < max_iter
    
    %% update Zk
    Z_pre = Z;
    R1 = J-Y2/beta;
    [Z,Z_nuc,Z_rank] = prox_tnn_w(R1,1/beta);
            
    %% update Lk  
      E_pre = E;
      %G2 = E -Y3/beta;  %+or-
      G2=(X-tprod(A,J)+Y1/beta+S+Y3/beta)/2;
      
     ImData2 = G2;  
     sizeImg = [size(ImData2,1),size(ImData2,3)];
     graph = getGraphSPAMS(sizeImg,[2,2]);
     D2 = ImData2;
     ImMean = mean(D2(:)); 
     D2 = D2 - ImMean; % subtract mean is recommended
     D3=Tensor_unfold(D2,2);
     L2 = solve_ls(D3,2*lambda/beta,graph);
     L4=fold_k(L2,2,size(G2));
          ImData2 = G2;      
     sizeImg = [size(ImData2,1),size(ImData2,2)];
     graph = getGraphSPAMS(sizeImg,[2,2]);
     D2 = ImData2;
     ImMean = mean(D2(:)); 
     D2 = D2 - ImMean; % subtract mean is recommended
     D3=Tensor_unfold(D2,3);
     L2 = solve_lstruct(D3,2*lambda/beta,graph);
     L5=fold_k(L2,3,size(G2));
          ImData2 = G2;      
     sizeImg = [size(ImData2,2),size(ImData2,3)];
     graph = getGraphSPAMS(sizeImg,[2,2]);
     D2 = ImData2;
     ImMean = mean(D2(:)); 
     D2 = D2 - ImMean; % subtract mean is recommended
     D3=Tensor_unfold(D2,1);
     L2 = solve_lstruct(D3,2*lambda/beta,graph);
     L6=fold_k(L2,1,size(G2));
     E=(L4+L5+L6)/3;               
    
    %% updata Yk  
    S_pre = S;
    U=E-Y3/beta;
    sigmas =sqrt(miu/beta);
    S = bm4d_1(1,U, sigmas);
    
    %% update Jk   //W
    J_pre=J; %J(k-1)
    Q1=Z+Y2/beta;
    Q2=X-E+Y1/beta;
    J=tprod(Ain, Q1+tprod(AT,Q2)); %J(k)
    
    %% print
    iter = iter+1;
    B=reshape(E, num, Dim)';
    r_new=sqrt(sum(B.^2,1));
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
    area_TLRR = sum((PF_40(1:end-1)-PF_40(2:end)).*(PD_40(2:end)+PD_40(1:end-1))/2);
    fprintf('iter = %d\n',iter);
    %% check convergence
    leq1 = X-tprod(A,J)-E;
    leq2 = Z-J;
    leq3 = S-E;
    
    leqm1 = max(abs(leq1(:)));
    leqm2 = max(abs(leq2(:)));
    leqm3 = max(abs(leq3(:)));
    
    difJ = max(abs(J(:)-J_pre(:)));
    difE = max(abs(E(:)-E_pre(:)));
    difZ = max(abs(Z(:)-Z_pre(:)));
    difS = max(abs(S(:)-S_pre(:))); 
    err(iter) = max([leqm1,leqm2,leqm3,difJ,difZ,difE,difS]);
    if err < tol
        break;
        iter
    end
    %% update Lagrange multiplier and  penalty parameter beta
    Y1 = Y1 + beta*leq1;
    Y2 = Y2 + beta*leq2;
    Y3 = Y3 + beta*leq3;
    beta = min(beta*rho,max_beta);
end

%%11
function [E] = solve_lstruct(W,lambda,graph)
 E = solve_ls(W,lambda,graph);


function [x] = solve_ls(w,lambda,graph)
% min lambda |x|_2 + |x-w|_2^2
% graph paramters
    graph_param.regul='graph';
    graph_param.lambda= lambda; % regularization parameter
    graph_param.num_threads=-1; % all cores (-1 by default)
    graph_param.verbose=false;   % verbosity, false by default
    graph_param.pos=false;       % can be used with all the other regularizations
    graph_param.intercept=false; % can be used with all the other regularizations
    x = mexProximalGraph(w,graph,graph_param);