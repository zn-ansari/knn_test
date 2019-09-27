%run this script to load data, and normalize data

load('hw1_mnist35.mat')
%show 4 training samples
subplot(2,2,1)
image(reshape(trainx(12,:),28,28)');
subplot(2,2,2)
image(reshape(trainx(992,:),28,28)');
subplot(2,2,3)
image(reshape(trainx(1012,:),28,28)');
subplot(2,2,4)
image(reshape(trainx(1112,:),28,28)');
%%normalize  data
trainx=double(trainx)/255;
testx=double(testx)/255;

n_train=length(trainy);%total number of training samples
n_test=length(testy);%total number of test samples

D=zeros(n_train);
for i=1:n_train
    for j=1:n_train
        D(i,j)=norm(trainx(i,:)-trainx(j,:));
    end
end




D_test=zeros(n_train,n_test);



for i=1:n_train
    for j=1:n_test
        D_test(i,j)=norm(trainx(i,:)-testx(j,:));
    end
end
disp('k=3');
k=3;
%training loss
[A,Index]=sort(D);

predicted_y_train=sign(sum(trainy(Index(1:k,:))));
train_loss=sum(predicted_y_train'~=trainy)/n_train;
disp('train_loss=')
disp(train_loss)

%test loss
[A,Index_test]=sort(D_test);
predicted_y_test=sign(sum(trainy(Index_test(1:k,:))));
test_loss=sum(predicted_y_test'~=testy)/n_test;
disp('test_loss=')
disp(test_loss)


%%%
disp('k=5');
k=5;
%training loss
[A,Index]=sort(D);

predicted_y_train=sign(sum(trainy(Index(1:k,:))));
train_loss=sum(predicted_y_train'~=trainy)/n_train;
disp('train_loss=')
disp(train_loss)

%test loss
[A,Index_test]=sort(D_test);
predicted_y_test=sign(sum(trainy(Index_test(1:k,:))));
test_loss=sum(predicted_y_test'~=testy)/n_test;
disp('test_loss=')
disp(test_loss)
