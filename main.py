# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 12:50:00 2019

@author: jbk48
"""

# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import argparse

import nsml
import numpy as np
import tensorflow as tf
import triplet_loss
import pickle
from nsml import DATASET_PATH

from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from my_applications.resnet import ResNet101
from keras.applications.resnet50 import ResNet50
from keras import regularizers
from keras.models import Sequential,Model
from keras.layers import *
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics.pairwise import euclidean_distances
from keras.applications.resnet50 import preprocess_input
from Pooling import RoiPooling, GeMPooling2D, Local_GeMPooling2D
from get_regions import rmac_regions, get_size_vgg_feat_map
from keras.utils.training_utils import multi_gpu_model
from diffussion import *
from re_ranking import ECN

K_ = 50
QUERYKNN = 5
R = 2000
alpha = 0.9


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)


def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        model.load_weights(file_path)
        print('model loaded!')

    def infer(queries, db):

        # Query 개수: 195
        # Reference(DB) 개수: 1,127
        # Total (query + reference): 1,322
        
        retrieval_results = {}
        bsize = 200

        print('Total # of batches :', int(len(queries)/bsize)+1)
              
        db_output = ["./db_query_squared_8.pkl","./db_reference_squared_8.pkl"]
        
        if os.path.exists(db_output[0]):
            print("exists")
            with open(db_output[0], 'rb') as f:
                query_vecs = pickle.load(f)
            with open(db_output[1], 'rb') as f:
                reference_vecs = pickle.load(f)
        else:
            print("non-exists")
            for batch_r in range(int(len(db)/bsize) + 1):
                
                references_b, reference_img = preprocess_query(db, batch_r, bsize)
        
                '''
                print('test data load queries {} query_img {} references {} reference_img {}'.
                      format(len(queries), len(query_img), len(references), len(reference_img)))
                '''
        
                references_b = np.asarray(references_b)
                reference_img = np.asarray(reference_img)
        
                reference_img = reference_img.astype('float32')
                ##reference_img /= 255
            
        
                # inference         
          
                # caching db output, db inference
                reference_vecs_b = model.predict([reference_img,reference_img,reference_img])
                
                if (batch_r == 0):
                    reference_vecs = reference_vecs_b
                    references = references_b
                
                else:
                    reference_vecs = np.concatenate((reference_vecs,reference_vecs_b), axis=0)
                    references = np.concatenate((references, references_b))
                    
                print(batch_r, 'th ref batch complete.')
                
    
            for batch in range(int(len(queries)/bsize) + 1):
                
                queries_b, query_img = preprocess_query(queries, batch, bsize)
                
                queries_b = np.asarray(queries_b)
                query_img = np.asarray(query_img)
                
                # print(batch, 'th query batch shape: ', query_img.shape)
                
                query_img = query_img.astype('float32')
                ##query_img /= 255
                
                # inference
                # print('inference start')
                query_vecs_b = model.predict([query_img,query_img,query_img])
                                        
                '''
                reference_vecs = model.predict([reference_img[0][np.newaxis,:],reference_img[0][np.newaxis,:],reference_img[0][np.newaxis,:]])
                for i in range(1,len(reference_img)):
                    reference_vecs_b = model.predict([reference_img[i][np.newaxis,:],reference_img[i][np.newaxis,:],reference_img[i][np.newaxis,:]])
                    reference_vecs = np.concatenate((reference_vecs,reference_vecs_b), axis=0)
                '''
                
                if (batch == 0):
                    query_vecs = query_vecs_b
                    queries_all = queries_b
                
                else:
                    query_vecs = np.concatenate((query_vecs,query_vecs_b), axis=0)
                    queries_all = np.concatenate((queries_all, queries_b))
                                
                print(batch, 'th query batch complete.')
            
            with open(db_output[0], 'wb') as f:
                pickle.dump(query_vecs,f)            
            with open(db_output[1], 'wb') as f:
                pickle.dump(reference_vecs,f)       
        
        '''
        X = reference_vecs
        Q = query_vecs
        '''
        '''
        sim  = np.dot(Q,X.T)
        del Q
        qsim = sim_kernel(sim)
        
        del sim
        
        sortidxs = np.argsort(-qsim, axis = 1)
        for i in range(len(qsim)):
            qsim[i,sortidxs[i,QUERYKNN:]] = 0
        
        qsim = sim_kernel(qsim)
        A = np.dot(X,X.T)
        del X
        W = sim_kernel(A)
        del A
        W = topK_W(W, K_)
        Wn = normalize_connection_graph(W)
        del W
        
        indices = fsr_rankR(qsim, Wn, alpha, R)
        '''
        '''
        ## re_ranking 
        q_g_dist = euclidean_distances(Q,X)
        q_q_dist = euclidean_distances(Q,Q)
        g_g_dist = euclidean_distances(X,X)
        
        del X
        del Q
        
        distances = re_ranking(q_g_dist, q_q_dist, g_g_dist,k1=20,k2=6)
        indices = np.argsort(distances, axis=1)
        '''
        '''
        distances = ECN(query_vecs,reference_vecs)
        indices = np.argsort(distances, axis=1)
        '''
        ## re_ranking 
        
        
        
        ## distances = euclidean_distances(query_vecs,reference_vecs)
        distances = np.dot(query_vecs,reference_vecs.T)
        indices = np.argsort(-distances, axis=1)
        
        ## query expansion
        for expand_epoch in range(1):
            query_vecs = np.vstack([np.vstack((query_vecs[i], reference_vecs[indices[i, :5]])).mean(axis=0) for i in range(len(query_vecs))])
            distances = np.dot(query_vecs,reference_vecs.T)
            indices = np.argsort(-distances, axis=1)
       
        
        for (i, query) in enumerate(queries_all):
            query = query.split('/')[-1].split('.')[0]
            ranked_list = [references[k].split('/')[-1].split('.')[0] for k in indices[i]]
            ranked_list = ranked_list[:1000]

            retrieval_results[query] = ranked_list
                    
                    
        print('done')

        return list(zip(range(len(retrieval_results)), retrieval_results.items()))

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)



def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm



# data preprocess
def preprocess_query(queries, index, size):
    query_img = []
    img_size = (224, 224)
    batch_queries = queries[size*index : min(size*(index+1), len(queries))]

    for img_path in batch_queries:
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255
        ##img = preprocess_input(img)
        query_img.append(img[0])

    return batch_queries, query_img

def addition(x):
    sum = K.sum(x, axis=1)
    return sum


def convnet_model_():
    vgg_model = ResNet50(weights=None, include_top=False, input_shape = (224,224,3))
    x = vgg_model.output
    x = GeMPooling2D()(x)
    x = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x)
    '''
    x = Dense(512, name='pca',
                              kernel_initializer='identity',
                              bias_initializer='zeros')(x)
    x = Lambda(lambda x: K.l2_normalize(x, axis=1), name='pca_norm')(x)
    '''
    convnet_model = Model(inputs=vgg_model.input, outputs=x)
    return convnet_model


def deep_rank_model():
 
    convnet_model = convnet_model_()
    first_input = Input(shape=(224,224,3))
    first_avg = AveragePooling2D(pool_size=(4,4), padding = 'same')(first_input)
    first_conv = Conv2D(96, kernel_size=(8, 8),strides=(4,4), padding='same',kernel_regularizer=regularizers.l2(0.01))(first_avg)
    first_max = MaxPool2D(pool_size=(7,7),strides = (4,4),padding='same')(first_conv)
    first_max = Flatten()(first_max)
    first_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(first_max)

    second_input = Input(shape=(224,224,3))
    second_avg = AveragePooling2D(pool_size=(8,8), padding = 'same')(second_input)
    second_conv = Conv2D(96, kernel_size=(8, 8),strides=(4,4), padding='same', kernel_regularizer=regularizers.l2(0.01))(second_avg)
    second_max = MaxPool2D(pool_size=(3,3),strides = (2,2),padding='same')(second_conv)
    second_max = Flatten()(second_max)
    second_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(second_max)

    merge_one = concatenate([first_max, second_max])

    merge_two = concatenate([merge_one, convnet_model.output])
    emb = Dense(2048)(merge_two)
    l2_norm_final = Lambda(lambda  x: K.l2_normalize(x,axis=1))(emb)

    final_model = Model(inputs=[first_input, second_input, convnet_model.input], outputs=l2_norm_final)

    return final_model


def generator_three_img(gen,dir1,target_size=(224,224), batch_size=32, class_mode="categorical"):
    genX1 = gen.flow_from_directory(dir1, target_size = target_size, batch_size=batch_size, 
                                    class_mode=class_mode, shuffle = True, seed=7)
    while True:
        X1i = genX1.next()
        label = np.argmax(X1i[1],axis=1)
        New_X = preprocess_input(X1i[0])
        yield [New_X,New_X,New_X], label


def generator_three_img_sampling(gen,dir1,target_size=(224,224), batch_size=20, class_mode="categorical"):

    train_dataset_path = DATASET_PATH + '/train/train_data'
    train_class_list = os.listdir(train_dataset_path)

    num_class = len(train_class_list)
    sub_batch = 5
    num_neg = 1
    num_class = int(batch_size/sub_batch)

    genX_list = []

    for train_class in train_class_list:
        genX = gen.flow_from_directory(dir1, classes=[train_class],target_size = target_size, batch_size=sub_batch, 
            class_mode=class_mode, shuffle = True, seed=7)
        genX_list.append(genX)


    while True:

        class_list = np.random.choice(range(num_class), num_class, replace = False)
        for n in range(len(class_list)):
            if n == 0:
                total_data = genX_list[class_list[n]].next()
                img_data = total_data[0]
                ##img_data = preprocess_input(total_data[0])
                label_data = np.array([class_list[n] for i in range(len(total_data[0]))])
            else:    
                total_data = genX_list[class_list[n]].next()
                img_data = np.concatenate((img_data, total_data[0]))
                ##img_data = preprocess_input(total_data[0])
                label_data = np.concatenate((label_data, np.array([class_list[n] for i in range(len(total_data[0]))])))

        yield [img_data, img_data, img_data], label_data


if __name__ == '__main__':
        
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--batch_size', type=int, default=150)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0', help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

    # training parameters
    nb_epoch = config.epochs
    batch_size = config.batch_size

    num_classes = 1383
    num_train_data = 73551
    input_shape = (224, 224, 3)  # input image shape
    

    deep_rank_model_ = deep_rank_model()
    deep_rank_model_.summary()
    

    bind_model(deep_rank_model_)
    

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        
        nsml.load(checkpoint='4', session='team_52/ir_ph2/367')
        nsml.save('saved')
        exit()
        
        
        """ Initiate RMSprop optimizer """
        deep_rank_model_.compile(loss=triplet_loss.triplet_semihard_loss, optimizer=Adam(lr=0.0001))

        """ Load data """
        print('dataset path', DATASET_PATH)
        ##nsml.load(checkpoint='1', session='team_52/ir_ph2/424')
        train_dataset_path = DATASET_PATH + '/train/train_data'
        train_class_list = os.listdir(train_dataset_path)

        if nsml.IS_ON_NSML:
            # Caching file
            train_datagen = ImageDataGenerator(rescale = 1./255,
                                               shear_range = 0.2,
                                               zoom_range = 0.2,
                                               horizontal_flip = True,
                                               width_shift_range=0.2,
                                               height_shift_range=0.2,
                                               rotation_range=30,
                                               fill_mode="nearest")
            
            training_set = generator_three_img_sampling(train_datagen, train_dataset_path, target_size= input_shape[:2], 
                                batch_size = batch_size, class_mode = 'categorical')
            
            
        
        steps_per_epoch = int(num_train_data/batch_size)
        
        for epoch in range(nb_epoch):
            res = deep_rank_model_.fit_generator(training_set,steps_per_epoch = steps_per_epoch, 
                                           initial_epoch=epoch,
                                           epochs = epoch+1,
                                           verbose=1,
                                           shuffle = True)
            
            print(res.history)
            train_loss = res.history['loss'][0]            
            nsml.report(summary=True, epoch=epoch, epoch_total=nb_epoch, loss=train_loss)
            nsml.save(epoch)