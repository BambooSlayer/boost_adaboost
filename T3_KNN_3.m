load fisheriris %导入matlab自带的植物学的150个样本，即n=150
data=meas(:,[3,4]);%取2个属性值用于分类
groups=2*ismember(species,'versicolor')+ismember(species,'setosa');
[train, test] = crossvalind('holdOut',groups,0.6);%随机选择训练集合测试集
data_train=meas(train,[1,3]);
groups_train=groups(train);
data_test=meas(test,[1,3]);
groups_test=groups(test);
%% （1）
N1=7;
%从大小为n（?）的原始样本集D中不放回得随机选取n1（?）个样本点，得到样本集D1
RAND=randperm(length(groups_train));%生成1~?的随机整数序列
D1=data_train(RAND(1:N1),:);
D1_species=groups_train(RAND(1:N1));
%根据D1训练第一个弱分类器C1。
% svmStruct1 = svmtrain(D1,D1_species,'showplot',true);%训练C1分类器
mdl1 = ClassificationKNN.fit(D1,D1_species,'NumNeighbors',3);
%使用svmtrain进行训练,得到训练后的结构svmStruct,在预测时使用.

%% （2）
unFL=RAND(N1+1:length(groups_train));%取未分类的RAND(N1+1:?,:)
classes1 =  predict(mdl1, data_train(unFL,:));%svmclassify(svmStruct1,data_train(unFL,:),'showplot',true);
%     hold on
%     plot(data(groups,1),data(groups,2),'r*')
COIN=randi([0,1],N1,1);%掷硬币N1次
TRUE=unFL(find(classes1==groups_train(unFL)));%因为classes1是随机排序的N1后的序列，所以加
TRUE_rand=randperm(length(TRUE));%打乱TRUE序号
TRUE_num=sum(COIN);%要取C1分正确的个数
D2=data_train(TRUE(TRUE_rand(1:TRUE_num)),:);
FAL=unFL(find(classes1~=groups_train(unFL)));
FAL_rand=randperm(length(FAL));
FAL_num=N1-TRUE_num;
D2=[D2;data_train(FAL(FAL_rand(1:FAL_num)),:)];
D2_species=groups_train([TRUE(TRUE_rand(1:TRUE_num)),FAL(FAL_rand(1:FAL_num))]);
%根据D2训练第一个弱分类器C1。
% figure(2)
% svmStruct2 = svmtrain(D2,D2_species,'showplot',true);%训练C2分类器
mdl2 = ClassificationKNN.fit(D2,D2_species,'NumNeighbors',3);
%     hold on
%     plot(data(groups,1),data(groups,2),'r*')

%% (3）
%D2: 
FLed=[TRUE(TRUE_rand(1:TRUE_num)),FAL(FAL_rand(1:FAL_num))];
%D1:  
FLed=[FLed,RAND(1:N1)];
unFL2=ones(1,length(groups_train));
unFL2(FLed)=0;
unFL2=find(unFL2==1);
% unFL=RAND(？？,:);%取未分类的RAND(N1+1:150,:)
classes3_1 =  predict(mdl1,data_train(unFL2,:));%svmclassify(svmStruct1,data_train(unFL2,:),'showplot',true);
classes3_2 = predict(mdl2,data_train(unFL2,:));%svmclassify(svmStruct2,data_train(unFL2,:),'showplot',true);
D3=data_train(unFL2(find((classes3_2~=classes3_1)==1)),:);
D3_species=groups_train(unFL2(find((classes3_2~=classes3_1)==1)));
% figure(3)
% svmStruct3 = svmtrain(D3,D3_species,'showplot',true);%训练C3分类器
mdl3 = ClassificationKNN.fit(D3,D3_species,'NumNeighbors',3);

%% (4)
figure()
% plot(data(groups,1),data(groups,2),'r.')
% hold on
% plot(data(groups==0,1),data(groups==0,2),'g.')
classes1 = predict(mdl1,data_test);%svmclassify(svmStruct1,data_test,'showplot',true);
%  plot(data(classes1,1),data(classes1,2),'ro','MarkerSize',5)
classes2 = predict(mdl2,data_test);%svmclassify(svmStruct2,data_test,'showplot',true);
%  plot(data(classes2,1),data(classes2,2),'bo','MarkerSize',7)
classes3 = predict(mdl3,data_test);%svmclassify(svmStruct3,data_test,'showplot',true);
%  plot(data(classes3,1),data(classes3,2),'go','MarkerSize',9)
classes=classes3;
C1_2=find((classes1 == classes2)==1);
classes(C1_2)=classes1(C1_2);
% legend('类别1','类别2','C1-类别1','C2-类别1','C3-类别1')
sum(groups_test==classes1)/length(classes)
sum(groups_test==classes2)/length(classes)
sum(groups_test==classes3)/length(classes)
sum(groups_test==classes)/length(classes)