function [w_ij0,w_ki0,gammai,betai,mui,varti,gammak,betak,muk,vartk,err_train]=train_net1(train_data,train_label,mini_batch,f1,max_epoch,lr,midnum,decay_epoch,sigma)
%输入：训练数据和标签；输出：训练好的参数
%初始化网络

%% 三层神经网络的创建
innum = size(train_data,1); %输入神经元个数(输入维数)
%midnum = 200; %隐层神经元个数
outnum = size(train_label,1); %输出神经元个数
% 权值初始化
w_ij0 = (rands(midnum, innum));
w_ki0 = (rands(outnum,midnum));
gammai=1.0000001;
gammak=1.0000001;
betai=0.0000001;
betak=0.0000001;

err_goal = 0.09; %误差目标最小值
%max_epoch = 1000; %最大训练次数
%mini_batch=500;
%lr=40;
lr0=lr;
[~,N] = size(train_data); 
Oi = zeros(midnum,N);Ok = zeros(outnum,N);  % 一个40行4500列的零矩阵，一个10行4500列的零矩阵
w_ij= zeros(midnum,innum);    % 40行35列的零矩阵
w_ki = zeros(outnum,midnum);% 10行40列的零矩阵
E0=inf;
r=0.5;
copies=N/mini_batch;
record_muk=zeros(outnum,max_epoch*copies);
record_mui=zeros(midnum,max_epoch*copies);
m={0,0,0,0,0,0};v={0,0,0,0,0,0};delta=cell(1,6);
%% training
for epoch = 1:max_epoch     % 运行max-epoch次
    k=rand(1,N);  
    [~,n]=sort(k);
    train_data=train_data(:,n(1:N));
    train_label=train_label(:,n(1:N));
    for i = 1:mini_batch:N-mini_batch+1 
        num=(i+mini_batch-1)/mini_batch;
     %计算隐层神经元节点输出
         w_ij = binary(w_ij0);%输入层到隐层的权值二值化  
         NETi(:,i:i+mini_batch-1) = w_ij*train_data(:,i:i+mini_batch-1);
         [Oi(:,i:i+mini_batch-1),mui,xmui,xhati,vari,varti] = BN_train(NETi(:,i:i+mini_batch-1),gammai,betai);%输出标准化
         Oib(:,i:i+mini_batch-1) = binary(Oi(:,i:i+mini_batch-1));%输出二值化
    %计算输出层神经元节点输出
        w_ki =binary(w_ki0);%隐层到输出层的权值二值化
        NETk(:,i:i+mini_batch-1) = w_ki*Oib(:,i:i+mini_batch-1);
        [Ok(:,i:i+mini_batch-1),muk,xmuk,xhatk,vark,vartk]=BN_train(NETk(:,i:i+mini_batch-1),gammak,betak);%输出标准化
        Ok(:,i:i+mini_batch-1)=1./(1+exp((-Ok(:,i:i+mini_batch-1))));  
    %调整输出层权重
    delta_k = (Ok(:,i:i+mini_batch-1)-train_label(:,i:i+mini_batch-1)).*Ok(:,i:i+mini_batch-1).*(1-Ok(:,i:i+mini_batch-1));
    [delta_NETk,dgammak,dbetak] = BBN(delta_k,xmuk,vark,gammak,xhatk);%标准化反向传播
    delta_i =  w_ki'*delta_NETk;
    delta_wk=delta_NETk*Oib(:,i:i+mini_batch-1)'/mini_batch;
    delta_i=delta_i.*(Oi(:,i:i+mini_batch-1)<=1 & Oi(:,i:i+mini_batch-1)>=-1);
    [delta_NETi,dgammai,dbetai] = BBN(delta_i,xmui,vari,gammai,xhati);
    delta_wi=delta_NETi*train_data(:,i:i+mini_batch-1)'/mini_batch;
    %Adam
    grad={delta_wk,delta_wi,dgammak,dgammai,dbetak,dbetai};
    for j=1:6
    [delta{j},m{j},v{j}]=adam(grad{j},m{j},v{j},epoch,num,copies);
    end
%    %更新存在误差
%    for j=1:6
%        [MM,NN]=size(delta{j});
%        delta{j}=delta{j}.*(1+(sigma/3)^2*randn(MM,NN));
%    end
    
    %更新参数
    w_ki00=clip(w_ki0-delta{1}*lr);
    w_ij00=clip(w_ij0-delta{2}*lr);
    w2=binary(w_ij00);delta_ww=w2-w_ij;
       w_ki0=w_ki00;w_ij0=w_ij00;
    gammak=gammak-delta{3}*lr;
    gammai=gammai-delta{4}*lr;
    
    betak=betak-delta{5}*lr;
    betai=betai-delta{6}*lr;
   
     %记录BN层均值和方差
     j=(epoch-1)*copies+num;
     record_muk(:,j)=muk(:);
     record_mui(:,j)=mui(:);
     record_vartk(:,j)=vartk(:);
     record_varti(:,j)=varti(:);
    end
    
    %计算误差函数
    E = 1/2*sum(sum((train_label-Ok).^2));
    err_train(1,epoch)=E;
    [~,h]=max(Ok);[t,~]=find(train_label==1);
    for j=1:N 
   if find(Ok(:,j)>=0.9)
       c=find(Ok(:,j)>=0.9);
       [~,d]=max(Ok(:,j));
       x(:,j)=intersect(c,d);
   else 
       x(1,j)=100;
   end
end
    accuracy = length(find(abs(x-t')==0))/length(Ok);

    if E<=err_goal
      break
    end
    %学习率衰减    
    decay_epoch=25;
    if mod(epoch,decay_epoch)==0
        lr=lr0*(0.7^(epoch/decay_epoch));
    end
    
    
    %制图
    %%str=['mini batch=',num2str(mini_batch)];
    %err(1,epoch)=E;
    %subplot(2,1,1)
    %scatter(epoch,E,'.',f1);
    %%title('训练误差(浮点值)');
    %%legend(str)
    %hold on
    %drawnow
    
    
    %acc(1,epoch)=accuracy;
    %subplot(2,1,2)
    %scatter(epoch,accuracy,'.',f1);
    %%title('训练情况(百分比)');
    %%legend(str);
    %hold on
    %drawnow
   
epoch; %显示训练次数
%fp = fopen('w_ij.txt','w+');
%fprintf(fp,'%f\t',w_ij);
%fclose(fp);
%fp = fopen('w_ki.txt','w+');
%fprintf(fp,'%f\t',w_ki);
%fclose(fp);
end
mui=mean(record_mui,2);
muk=mean(record_muk,2);
varti=mean(record_varti,2);
vartk=mean(record_vartk,2);