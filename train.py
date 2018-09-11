import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import h5py
import numpy as np
import cv2
from batchup import data_source
from PIL import Image
import io
import torch.utils.data as Data
import torch
from torchnet.meter import AverageValueMeter, MovingAverageValueMeter

from model.faster_rcnn import faster_rcnn
from model.utils.transform_tools import image_normalize
from chainercv.visualizations import vis_bbox
torch.manual_seed(1)
epoch_limit = 5

filepath='/home/mayank-s/PycharmProjects/Datasets/Berkely_DeepDrive/frcnn_model_parameter/mytraining.pt'

# _________________________________________________________________________________________
voc_path = '/home/mayank-s/PycharmProjects/Datasets/Berkely_DeepDrive'

train_image_path='/home/mayank-s/PycharmProjects/Datasets/Berkely_DeepDrive/bdd100k/images/100k/train'

print('Reading HDF5 dataset structure.')
fname = os.path.join(voc_path, 'berkely_datasets.hdf5')
my_data=h5py.File(fname,'r')
at_item = my_data.attrs.values()#reading attributes
# print(list(at_item))
val=my_data.get('train')
image_data=val.get('images')
image_return=np.array(image_data)
# print(image_return[0])

box_return=val.get('boxes')
box_data=list(box_return)
# box_return=np.array(bbox_data)
print("single box shape",box_return[0].shape)
print("total box shape",box_return.shape)
my_data.close()
print("closing hdf5 file")


def adjust_learning_rate(optimizer, epoch, init_lr, lr_decay_factor=0.1, lr_decay_epoch=10):
    """Sets the learning rate to the initial LR decayed by lr_decay_factor every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch == 0:
        lr = init_lr * (lr_decay_factor ** (epoch // lr_decay_epoch))
        print('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



model = faster_rcnn(20, backbone='vgg16')
if torch.cuda.is_available():
    model = model.cuda()

optimizer = model.get_optimizer(is_adam=False)
avg_loss = AverageValueMeter()
ma20_loss = MovingAverageValueMeter(windowsize=20)
model.train()


for epoch in range(epoch_limit):
    adjust_learning_rate(optimizer, epoch, 0.001, lr_decay_epoch=10)
    for loop, bbox_val in enumerate(box_data):
        img = Image.open(io.BytesIO(image_return[loop]))
        img.save("temp.jpg")
        image = cv2.imread("temp.jpg", 1)
        image_expanded = np.expand_dims(image, axis=0)
        array_data = np.array(bbox_val)
        array_data = array_data.reshape(array_data.shape[0], 1)
        box_val = array_data.reshape(-1, 5)
        my_label = box_val[:, 0]
        my_bbox = box_val[:, 1:5]
        # print(my_bbox.shape)
        # print(my_label.shape)
        # print(image)
        image = np.swapaxes(image, 0, 2)
        image = np.swapaxes(image, 1, 2)
        image=(image/255)

        loss = model.loss(image, my_bbox, my_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.cpu().data.numpy()
        avg_loss.add(loss_value)
        ma20_loss.add(float(loss_value))
        print('[epoch:{}]  [batch:{}]  [sample_loss:{:.4f}]  [avg_loss:{:.4f}] '.format(epoch, loop, loss_value, avg_loss.value()[0]))
    # torch.save(model.state_dict(), filepath)#saving model parameter


model.eval()
# model.load_state_dict(torch.load(filepath))
my_data=h5py.File(fname,'r')
at_item = my_data.attrs.values()#reading attributes
# print(list(at_item))
val=my_data.get('val')
image_data=val.get('images')
image_return=np.array(image_data)
# print(image_return[0])
my_data.close()
print("closing hdf5 file")

for loop,image_dt in enumerate (image_return):
    img = Image.open(io.BytesIO(image_dt))
    img.save("temp.jpg")
    image = cv2.imread("temp.jpg", 1)
    image=np.swapaxes(image,0,2)
    image=np.swapaxes(image,1,2)
    imgx=(image/255)
    # imgx = img/255
    bbox_out, class_out, prob_out = model.predict(imgx, prob_threshold=0.95)

    vis_bbox(img, bbox_out, class_out, prob_out,label_names="predicted boxes")
    plt.show()
    fig = plt.gcf()
    fig.set_size_inches(11, 5)
    #fig.savefig('test_'+str(i)+'.jpg', dpi=100)

