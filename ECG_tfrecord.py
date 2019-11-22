# -*- coding: utf-8 -*-
"""
@author: caokai
"""

import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#cwd='./MI_image/7_fold/fold3/test/'
cwd = './fuxian_image/train/'
classes = {'0', '1'}  
#writer= tf.python_io.TFRecordWriter("MI_test3.tfrecords") 
writer = tf.python_io.TFRecordWriter("fuxian_train.tfrecords")  


for index, name in enumerate(classes):
    class_path = cwd + name + '/'
    print(class_path)
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name  
        print(img_name[0:9])
        img = Image.open(img_path)
        ########
        txt_path =  "MI_image/7_fold/txt/"+ img_name[0:9] +".txt"
        f = open(txt_path)
        line = f.read()
        line_split = line.split("\t")
        line_split = line_split[1:]
        line_split = line_split[:-1]
        line_split = list(map(int,line_split))
        img1 = img.crop((line_split[0], line_split[1], line_split[2], line_split[3]))
        img2 = img.crop((line_split[4], line_split[5], line_split[6], line_split[7]))
        img3 = img.crop((line_split[8], line_split[9], line_split[10], line_split[11]))
        img4 = img.crop((line_split[12], line_split[13], line_split[14], line_split[15]))
        img5 = img.crop((line_split[16], line_split[17], line_split[18], line_split[19]))
        img6 = img.crop((line_split[20], line_split[21], line_split[22], line_split[23]))
        img7 = img.crop((line_split[24], line_split[25], line_split[26], line_split[27]))
        img8 = img.crop((line_split[28], line_split[29], line_split[30], line_split[31]))
        img9 = img.crop((line_split[32], line_split[33], line_split[34], line_split[35]))
        img10 = img.crop((line_split[36], line_split[37], line_split[38], line_split[39]))
        img11 = img.crop((line_split[40], line_split[41], line_split[42], line_split[43]))
        img12 = img.crop((line_split[44], line_split[45], line_split[46], line_split[47]))
        img = img.resize((128, 128))
        img1 = img1.resize((128, 128))
        img2 = img2.resize((128, 128))
        img3 = img3.resize((128, 128))
        img4 = img4.resize((128, 128))
        img5 = img5.resize((128, 128))
        img6 = img6.resize((128, 128))
        img7 = img7.resize((128, 128))
        img8 = img8.resize((128, 128))
        img9 = img9.resize((128, 128))
        img10 = img10.resize((128, 128))
        img11 = img11.resize((128, 128))
        img12 = img12.resize((128, 128))
        img_raw = img.tobytes()  
        address_raw =  img_name[0:9].encode()
        img_raw1 = img1.tobytes()
        img_raw2 = img2.tobytes()
        img_raw3 = img3.tobytes()
        img_raw4 = img4.tobytes()
        img_raw5 = img5.tobytes()
        img_raw6 = img6.tobytes()
        img_raw7 = img7.tobytes()
        img_raw8 = img8.tobytes()
        img_raw9 = img9.tobytes()
        img_raw10 = img10.tobytes()
        img_raw11 = img11.tobytes()
        img_raw12 = img12.tobytes()
        ####
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            "address_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[address_raw])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            ####
            'img_raw1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw1])),
            'img_raw2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw2])),
            'img_raw3': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw3])),
            'img_raw4': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw4])),
            'img_raw5': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw5])),
            'img_raw6': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw6])),
            'img_raw7': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw7])),
            'img_raw8': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw8])),
            'img_raw9': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw9])),
            'img_raw10': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw10])),
            'img_raw11': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw11])),
            'img_raw12': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw12]))
            ####
        }))  # example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  # 序列化为字符串

writer.close()

# import os
# import tensorflow as tf
# from PIL import Image
# import sys


# def creat_tf(imgpath):
#     classes = os.listdir(imgpath)

#     writer = tf.python_io.TFRecordWriter("zengqiang_fuxian_train_224.tfrecords")

#     for index, name in enumerate(classes):
#         class_path = imgpath + name
#         if os.path.isdir(class_path):
#             for img_name in os.listdir(class_path):
#                 img_path = class_path + '/' + img_name
#                 img = Image.open(img_path)
#                 # you can improve, not resize
#                 img = img.resize((224, 224))
#                 img_raw = img.tobytes()
#                 example = tf.train.Example(features=tf.train.Features(feature={
#                     'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(name)])),
#                     'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
#                 }))
#                 writer.write(example.SerializeToString())
#     writer.close()


def read_example():
    for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        # image = example.features.feature['img_raw'].bytes_list.value
        label = example.features.feature['label'].int64_list.value
        print(label[0])


if __name__ == '__main__':
    imgpath = 'zengqiang_fuxian_image_zonghe/train/'
    creat_tf(imgpath)
    # read_example() 
 
