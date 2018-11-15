function predict_label=test_net1(test_data,w_ij0,w_ki0,gammai,betai,mui,varti,gammak,betak,muk,vartk)
%根据训练好的权值w_ij,w_ki,对给定的输入计算输出
%test_data,test_label
%innum = 35; %输入神经元个数
%midnum = 50; %隐层神经元个数
%outnum = 10; %输出神经元个数
%计算隐层输出
    w_ij = binary(w_ij0);%输入层到隐层的权值二值化  
    NETi = w_ij*test_data;
    [Oi]=BN_test(NETi,gammai,betai,mui,varti);
    Oib = binary(Oi);
%计算输出层神经元输出
    w_ki =binary(w_ki0);%隐层到输出层的权值二值化
    NETk = w_ki*Oib;
    [Ok]=BN_test(NETk,gammak,betak,muk,vartk);
    Ok=1./(1+exp((-Ok)));
predict_label = Ok;