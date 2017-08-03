clc
clear
%% （1）导入数据
load fisheriris %导入matlab自带的植物学的150个样本，即n=150
data=meas(:,[2,4]);%取2个属性值用于分类
groups=ismember(species,'versicolor');%把分类简化成两类：??和non-??
[train, test] = crossvalind('holdOut',groups,0.5);%随机选择训练集合测试集
data_train=meas(train,[1,3]);
groups_train=groups(train);
data_test=meas(test,[1,3]);
groups_test=groups(test);
%% （2)初始化权重
l=sum(groups_train);%the number of positives
m=length(groups_train)-l;%the number of negatives
i=find(groups_train==0);
j=find(groups_train==1);
w=zeros(1,length(groups_train));
w(1,i)=1/(2*m);
w(1,j)=1/(2*l);
w(1,:)=w(1,:)/sum(w(1,:));% normalize the weights     
t=1;
%% 训练弱分类器
while 1
    N1=randint(1,1,5)+5;%选择5~9个随机数据点
    while 1
        RAND=randperm(length(groups_train));%生成1~?的随机整数序列            
        D1_species=groups_train([find(w(t,:)==max(w(t,:))),RAND(1:N1)]);
        if   mod(sum(D1_species),length(D1_species))~=0 %如果随机挑选的样本中包含两类，则继续
            break
        end
    end
    D1=data_train([find(w(t,:)==max(w(t,:))),RAND(1:N1)],:);
    svmStruct_j(t) = svmtrain(D1,D1_species,'showplot',true);%训练弱分类器
    classes_j(:,t) = svmclassify(svmStruct_j(t),data_train);
    ej=sum(w(t,:)'.*abs(classes_j(:,t)-groups_train));
    for k=1:100 % choose the classifier
        while 1
            RAND=randperm(length(groups_train));%生成1~?的随机整数序列            
            D1_species=groups_train([find(w(t,:)==max(w(t,:))),RAND(1:N1)]);
            if   mod(sum(D1_species),length(D1_species))~=0 %如果随机挑选的样本中包含两类，则继续
                break
            end
        end
        D1=data_train([find(w(t,:)==max(w(t,:))),RAND(1:N1)],:);
        svmStruct_j2 = svmtrain(D1,D1_species);%训练分类器
        classes_j2 = svmclassify(svmStruct_j2,data_train);
        ej2=sum(w(t,:)'.*abs(classes_j2-groups_train));
        if ej2<ej
            svmStruct_j(t)=svmStruct_j2;
            classes_j(:,t) = classes_j2 ;
            ej=ej2;
        end
    end
    e(t)=ej;
    tf=abs(classes_j(:,t)-groups_train);
    b(t)=e(t)/(1-e(t));
    for i=1:length(groups_train)%更新权重
        w(t+1,i)=w(t,i)*(b(t))^(1-tf(i));% normalize the weights       
    end
    a=log(1./b);
    A=(a*classes_j')>=(0.5*sum(a));
    P(t)=sum(A' == groups_train)/length(A)
    if P(t)>0.9
        break
    end
    t=t+1
    w(t,:)=w(t,:)/sum(w(t,:));% normalize the weights       
end
T=t;
a=log(1./b);
A=(a*classes_j')>=(0.5*sum(a));
clc 
 %% 对测试集进行测试
for t=1:T
    figure()
    classes_test(:,t) = svmclassify(svmStruct_j(t),data_test);
    per(t)=sum(classes_test(:,t) == groups_test)/length(groups_test);
    
end
A2=(a*classes_test')>=(0.5*sum(a));
P
per
PER=sum(A2' == groups_test)/length(A2)

 