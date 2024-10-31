import gzip
import os
import time
import random
import struct
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset
from paddle.vision.transforms import Compose, Normalize, ToTensor
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

class FashionData(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, dir='./', type='train', trans=None):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        """
        backend = paddle.vision.get_image_backend()
        if backend not in ['pil', 'cv2']:
            raise ValueError(
                "Expected backend are one of ['pil', 'cv2'], but got {}"
                    .format(backend))
        self.backend = backend
        self.type = type.lower()
        self.images_data_dir = os.path.join(dir, '%s-images-idx3-ubyte.gz' % type)
        self.labels_data_dir = os.path.join(dir, '%s-labels-idx1-ubyte.gz' % type)
        self.trans = trans
        # 将数据集读取到内存中
        self._parse_dataset()
        self.dtype = paddle.get_default_dtype()

    def _parse_dataset(self, buffer_size=100):
        self.images = []
        self.labels = []
        with gzip.GzipFile(self.images_data_dir, 'rb') as image_file:
            img_buf = image_file.read()
            with gzip.GzipFile(self.labels_data_dir, 'rb') as label_file:
                lab_buf = label_file.read()
                step_label = 0
                offset_img = 0
                # 从大端读取
                # 从magic byte中获取文件信息
                # 图像文件：16字节
                magic_byte_img = '>IIII'
                magic_img, image_num, rows, cols = struct.unpack_from(magic_byte_img, img_buf, offset_img)
                offset_img += struct.calcsize(magic_byte_img)
                offset_lab = 0
                # 标签文件：8字节
                magic_byte_lab = '>II'
                magic_lab, label_num = struct.unpack_from(magic_byte_lab, lab_buf, offset_lab)
                offset_lab += struct.calcsize(magic_byte_lab)
                while True:
                    if step_label >= label_num:
                        break
                    fmt_label = '>' + str(buffer_size) + 'B'
                    labels = struct.unpack_from(fmt_label, lab_buf, offset_lab)
                    offset_lab += struct.calcsize(fmt_label)
                    step_label += buffer_size
                    fmt_images = '>' + str(buffer_size * rows * cols) + 'B'
                    images_temp = struct.unpack_from(fmt_images, img_buf, offset_img)
                    images = np.reshape(images_temp, (buffer_size, rows * cols)).astype('float32')
                    offset_img += struct.calcsize(fmt_images)
                    for i in range(buffer_size):
                        self.images.append(images[i, :])
                        self.labels.append(np.array([labels[i]]).astype('int64'))

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        image = np.reshape(image, [28, 28])
        if self.backend == 'pil':
            image = Image.fromarray(image.astype('uint8'), mode='L')
        if self.trans is not None:
            image = self.trans(image)
        if self.backend == 'pil':
            return image, label.astype('int64')
        return image.astype(self.dtype), label.astype('int64')

    def __len__(self):
        return len(self.labels)

# 数据标准化和增强
trans = Compose([Normalize(mean=[127.5], std=[127.5], data_format='CHW'), ToTensor()])
fashion_mnist_train_dataset = FashionData(dir='data/fashion', type="train", trans=trans)
fashion_mnist_test_dataset = FashionData(dir='data/fashion', type='t10k', trans=trans)

# 计算像素值的均值和方差
train_images = np.array(fashion_mnist_train_dataset.images)
mean_pixel_value = np.mean(train_images)
std_pixel_value = np.std(train_images)
print(f'像素值均值: {mean_pixel_value}, 像素值标准差: {std_pixel_value}')

# 绘制像素值分布的直方图
plt.figure(figsize=(10, 5))
plt.hist(train_images.reshape(-1), bins=50, color='gray', edgecolor='black')
plt.title('Histogram of pixel value distribution')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')
plt.show()

# 绘制像素值分布箱线图（取样100张图像）
sample_images = train_images[:100]

plt.figure(figsize=(10, 5))
sns.boxplot(data=sample_images.reshape(100, -1))
plt.title('Boxplot of pixel value distribution (sample of 100 images)')
plt.show()

# 正态性检验
k2, p = stats.normaltest(train_images.reshape(-1))
print(f'正态性检验p值: {p}')

# PCA降维
pca = PCA(n_components=50)
train_images_pca = pca.fit_transform(train_images)
test_images_pca = pca.transform(np.array(fashion_mnist_test_dataset.images))
print(f'PCA后训练数据形状: {train_images_pca.shape}')
print(f'PCA后测试数据形状: {test_images_pca.shape}')

# 定义多层感知机模型
class MultilayerPerceptron(nn.Layer):
    def __init__(self):
        super(MultilayerPerceptron, self).__init__()
        self.linear1 = nn.Linear(1*28*28, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 10)

    def forward(self, inputs):
        x = paddle.flatten(inputs, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        y = F.softmax(x, axis=1)
        return y

# 定义网络结构
network = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.Softmax(axis=1)
)

# 配置模型
model = paddle.Model(network)
model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()),
              loss=paddle.nn.CrossEntropyLoss(),
              metrics=paddle.metric.Accuracy())

model.summary((1, 28, 28))

# 训练模型
model.fit(fashion_mnist_train_dataset, epochs=5, batch_size=256, verbose=1)

# 评估模型
eval_result = model.evaluate(fashion_mnist_test_dataset, verbose=1)
print(eval_result)

# 预测测试集
predict_result = model.predict(fashion_mnist_test_dataset)

# 定义标签列表
label_list = ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

# 抽样展示
indexs = [2, 15, 38, 211, 222, 323]
for idx in indexs:
    print(f'第{idx}条记录 真实值： {fashion_mnist_test_dataset[idx][1][0]}   预测值：{np.argmax(predict_result[0][idx])}')

# 保存模型
model.save('inference_model', training=False)

# 计算混淆矩阵
true_labels = [label[1][0] for label in fashion_mnist_test_dataset]
predicted_labels = [np.argmax(pred) for pred in predict_result[0]]
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# 打印分类报告
class_report = classification_report(true_labels, predicted_labels, target_names=label_list)
print('分类报告:\n', class_report)

# 绘制混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_list, yticklabels=label_list)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
