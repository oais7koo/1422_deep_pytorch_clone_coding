#!/usr/bin/env python
# coding: utf-8

# # 합성곱 신경망 맛보기

# - 심층 신경망 비교를 위해서 일단 심층신경망 생성(ConvNet이 적용안된 것)
# - fashion_mnist 데이터셋은 28*28 크기의 그레이 이미지 7만개로 구성
# - label은 0에서 9까지의 정수값을 가지는 배열

# In[50]:


import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[51]:


# 이미지 내려받기
train_dataset = torchvision.datasets.FashionMNIST(
    root='psdata/ps5060', download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_dataset = torchvision.datasets.FashionMNIST(
    root='psdata/ps5060', download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))


# In[52]:


# 데이터를 메모리에 로딩하기
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50)


# In[53]:


train_dataset


# In[54]:


train_dataset[0].__class__


# In[55]:


train_dataset[0]  # 튜플로 되어 있는데 첫번째 요소는 이미지 두번째는 레이블


# In[56]:


train_dataset[0][0]


# In[57]:


train_dataset[0][1]


# In[58]:


len(train_dataset[0])  # 이미지와 레이블을 함께 받아오는 것이기 때문에 2개


# In[59]:


len(train_dataset[0][0])


# In[60]:


train_dataset[0][0].shape  # 그레이 이미지


# In[61]:


test_dataset  # 총 7만장중 6만장은 학습 1만장은 테스트


# In[62]:


# 분류에 사용될 클래스 정의
labels_map = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress',
              4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

fig = plt.Figure(figsize=(8, 8))
columns = 4
rows = 5
for i in range(1, columns * rows + 1):
    img_xy = np.random.randint(len(train_dataset))
    img = train_dataset[img_xy][0][0, :, :]  # 3차원 배열 생성
    fig.add_subplot(rows, columns, i)
    plt.title(labels_map[train_dataset[img_xy][1]])
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()


# # 심층심경망 모델

# In[63]:


# 심층 신경망 모델생성
class FashionDNN(nn.Module):  # nn은 딥러닝 모델(네트워크) 구성에 필요한 모듈이 모여있는 패키지
    def __init__(self):  # 클래스형태의 모델은 항상 torch.nn.Module을 상속받아야 한다.
        # 객체가 갖는 속성 값 초기화, super(FashionDNN) 은 FashionDNN의 부모 클래스(super)의 클래스를 상속받겠다는 것임
        super(FashionDNN, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=256)
        # 0.25만큼 텐서의 값이 0이됨, 0이 안되는 갓음 기존값의 1/(1-0.25)만큼 곱해져서 커짐
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, input_data):  # 순전파 함수. 이름은 반드시 forward로 지정해야함
        # view는 넘파이의 reshape 역할로 텐서 크기 변경 (-1, 784)은 (?, 784)의 크기로 변경(이차원 텐서로)
        out = input_data.view(-1, 784)
        out = F.relu(self.fc1(out))
        out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


# In[64]:


# 심층 신경망에서 필요한 파라미터 정의
learning_rate = 0.001
model = FashionDNN()
model.to(device)

criterion = nn.CrossEntropyLoss()  # 분류문제에서 사용하는 손실 함수
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate)  # 최적화 함수 경사하강법은 Adam 사용
print(model)


# In[69]:


# 심층 신경망에 데이터 적용
num_epochs = 5
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

predictions_list = []
labels_list = []

for epoch in range(num_epochs):
    for images, labels in train_loader:  # for를 이용해서 레코드를 하나씩 가지고 옴
        images, labels = images.to(device), labels.to(device)

        train = Variable(images.view(100, 1, 28, 28))  # 100은 배치 수인듯
        labels = Variable(labels)

        outputs = model(train)  # 학습데이터를 모델에 적용
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1

        if not (count % 50):  # 50으로 나누었을 때 나머지가 0 이면
            total = 0
            correct = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(
                    device)  # 모델이 데이터를 처리하기 위해서는 동일한 device에 있어야 함
                labels_list.append(labels)
                # autograd는 자동미분을 수행하는 파이토치 핵심 패키지로 자동 미분에 대한 값을 저장하기 위해서 tape를 사용
                test = Variable(images.view(100, 1, 28, 28))
                # 순전파 단계에서 테이브는 수행하는 모든 연산을 저장함(그런데 이게 설명이 맞나)
                # autograd는 Variable을 사용해서 역전파를 위한 미분값을 자동으로 계산해줌
                # 자동미분을 게산하기 위해서는 torch.augograd 패키지 안에 있는 variable를 이용해야 동작함
                outputs = model(test)
                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()
                total += len(labels)

            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)

        if not (count % 500):  # 500만다
            print("Iteration: {}, Loss: {}, Accuracy: {}".format(
                count, loss.data, accuracy))


# In[68]:


len(train_loader)  # 배치 수?
