function predict_label=test_net1(test_data,w_ij0,w_ki0,gammai,betai,mui,varti,gammak,betak,muk,vartk)
%����ѵ���õ�Ȩֵw_ij,w_ki,�Ը���������������
%test_data,test_label
%innum = 35; %������Ԫ����
%midnum = 50; %������Ԫ����
%outnum = 10; %�����Ԫ����
%�����������
    w_ij = binary(w_ij0);%����㵽�����Ȩֵ��ֵ��  
    NETi = w_ij*test_data;
    [Oi]=BN_test(NETi,gammai,betai,mui,varti);
    Oib = binary(Oi);
%�����������Ԫ���
    w_ki =binary(w_ki0);%���㵽������Ȩֵ��ֵ��
    NETk = w_ki*Oib;
    [Ok]=BN_test(NETk,gammak,betak,muk,vartk);
    Ok=1./(1+exp((-Ok)));
predict_label = Ok;