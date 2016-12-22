
# coding: utf-8

# In[1]:

#conv Neural Network
# tensorboard --logdir=/home/ncc/notebook/learn/tensorboard/log

import numpy as np 
import tensorflow as tf

import math
import time
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import os 


sess = tf.InteractiveSession()
test_img=np.load(file_locate+'test_img.npy');
print np.shape(test_img)
img_row = np.shape(test_img)[1]
img_col = np.shape(test_img)[2]

batch_size=30
print img_row ,img_col
n_classes =2
in_ch =3
out_ch1=512
out_ch2=512
out_ch3=512
out_ch4=512
out_ch5=512
out_ch6=512
out_ch7=512
out_ch8=512
out_ch9=512
out_ch10=512
out_ch11=512
out_ch12=512
out_ch13=512
fully_ch1=4096
fully_ch2 =4096

x= tf.placeholder("float",shape=[None,img_col , img_row , 3],  name = 'x-input')
y_=tf.placeholder("float",shape=[None , n_classes] , name = 'y-input')
keep_prob = tf.placeholder("float")

x_image= tf.reshape(x,[-1,img_row,img_col,3])

iterate=300000




weight_row =3 ; weight_col=3

pooling_row_size1=int(img_row/2)
pooling_row_size2=int(pooling_row_size1/2)
pooling_row_size3=int(pooling_row_size2/2)
pooling_row_size4=int(pooling_row_size3/2)
pooling_row_size5=int(pooling_row_size4/2)
pooling_col_size1=int(img_col/2)
pooling_col_size2=int(pooling_col_size1/2)
pooling_col_size3=int(pooling_col_size2/2)
pooling_col_size4=int(pooling_col_size3/2)
pooling_col_size5=int(pooling_col_size4/2)

print img_col , img_row


# In[2]:

with tf.device('/gpu:3'):
    #with tf.device('/gpu:1'):
    train_img=np.load(file_locate+'train_img.npy');
    train_lab=np.load(file_locate+'train_lab.npy');
    val_img= np.load(file_locate+'val_img.npy');
    val_lab = np.load(file_locate+'val_lab.npy');
    test_img=np.load(file_locate+'test_img.npy');
    test_lab=np.load(file_locate+'test_lab.npy');

    print "Training Data",np.shape(train_img)
    print "Training Data Label",np.shape(train_lab)
    print "Test Data Label",np.shape(test_lab)
    print "val Data Label" , np.shape(val_img)

    n_train= np.shape(train_img)[0]
    n_train_lab = np.shape(train_lab)[0]


# In[3]:

"""def weight_variable(name,shape):
    #initial = tf.truncated_normal(shape , stddev=0.1)
    initial = tf.get_variable(name,shape=shape , initializer = tf.contrib.layers.xavier_initializer())
    return tf.Variable(initial)"""
with tf.device('/gpu:2'):
    def bias_variable(shape):
        initial = tf.constant(0.1 , shape=shape)
        return tf.Variable(initial)



# In[4]:

with tf.device('/gpu:2'):
    def next_batch(batch_size , image , label):

        a=np.random.randint(np.shape(image)[0] -batch_size)
        batch_x = image[a:a+batch_size,:]
        batch_y= label[a:a+batch_size,:]
        return batch_x, batch_y


# In[5]:

with tf.device('/gpu:3'):

    def conv2d(x,w):
        return tf.nn.conv2d(x,w, strides = [1,1,1,1], padding='SAME')
    def max_pool_2x2(x):
        return tf.nn.max_pool(x , ksize=[1,2,2,1] ,strides = [1,2,2,1] , padding = 'SAME')


# In[6]:

with tf.device('/gpu:2'):
    
    w_conv1 = tf.get_variable("W1",[weight_row,weight_col,3,out_ch1] , initializer = tf.contrib.layers.xavier_initializer())
    b_conv1 = bias_variable([out_ch1])

    w_conv2 = tf.get_variable("W2",[weight_row,weight_col,out_ch1,out_ch2] , initializer = tf.contrib.layers.xavier_initializer())
    b_conv2= bias_variable([out_ch2])

    w_conv3 = tf.get_variable("W3" ,[weight_row,weight_col,out_ch2,out_ch3] , initializer = tf.contrib.layers.xavier_initializer())
    b_conv3 = bias_variable([out_ch3])

    w_conv4 =tf.get_variable("W4" ,[weight_row,weight_col,out_ch3,out_ch4] , initializer = tf.contrib.layers.xavier_initializer())
    b_conv4 = bias_variable([out_ch4])
with tf.device('/gpu:3'):
    w_conv5 = tf.get_variable("W5",[weight_row,weight_col,out_ch4,out_ch5] , initializer = tf.contrib.layers.xavier_initializer())
    b_conv5 = bias_variable([out_ch5])
    
    w_conv6= tf.get_variable("W6" , [weight_row , weight_col ,out_ch5, out_ch6] ,initializer = tf.contrib.layers.xavier_initializer())
    b_conv6 = bias_variable([out_ch6])

    w_conv7 = tf.get_variable("W7" , [weight_row , weight_col ,out_ch6 , out_ch7 ], initializer = tf.contrib.layers.xavier_initializer())
    b_conv7 = bias_variable([out_ch7])
    
    w_conv8 = tf.get_variable("W8" , [weight_row , weight_col ,out_ch7 , out_ch8 ] , initializer = tf.contrib.layers.xavier_initializer())
    b_conv8 = bias_variable([out_ch8])
    
    w_conv9 = tf.get_variable("W9" ,[weight_row , weight_col ,out_ch8 , out_ch9] , initializer = tf.contrib.layers.xavier_initializer())
    b_conv9 = bias_variable([out_ch9])
with tf.device('/gpu:2'):
    w_conv10 = tf.get_variable("W10" ,[weight_row ,weight_col ,out_ch9  ,out_ch10] , initializer = tf.contrib.layers.xavier_initializer())
    b_conv10 = bias_variable([out_ch10])
    
    w_conv11 = tf.get_variable("W11" ,[weight_row , weight_col ,out_ch10 , out_ch11 ] , initializer = tf.contrib.layers.xavier_initializer())
    b_conv11 = bias_variable([out_ch11])
                               
    w_conv12 = tf.get_variable("W12" , [ weight_row , weight_col , out_ch11 , out_ch12 ] ,initializer = tf.contrib.layers.xavier_initializer())
    b_conv12 =bias_variable([out_ch12])
                               
    w_conv13 = tf.get_variable("W13" , [weight_row , weight_col ,out_ch12 , out_ch13] , initializer = tf.contrib.layers.xavier_initializer())
    b_conv13 = bias_variable([out_ch13])
    


# In[7]:

#conncect hidden layer 
with tf.device('/gpu:2'):
    h_conv1 = tf.nn.relu(conv2d(x_image , w_conv1)+b_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_conv1 , w_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    h_conv3 = tf.nn.relu(conv2d(h_pool2 , w_conv3)+b_conv3)
    h_conv4 = tf.nn.relu(conv2d(h_conv3 , w_conv4)+b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)

    h_conv5 = tf.nn.relu(conv2d(h_pool4, w_conv5)+b_conv5)
    h_conv6= tf.nn.relu(conv2d(h_conv5 , w_conv6)+b_conv6)
    h_conv7= tf.nn.relu(conv2d(h_conv6 , w_conv7)+ b_conv7)
    h_pool7 = max_pool_2x2(h_conv7)
with tf.device('/gpu:3'):
    h_conv8 = tf.nn.relu(conv2d(h_pool7 , w_conv8)+b_conv8)              
    h_conv9= tf.nn.relu(conv2d(h_conv8 , w_conv9)+b_conv9)
    h_conv10= tf.nn.relu(conv2d(h_conv9 , w_conv10)+b_conv10)
    h_pool10 = max_pool_2x2(h_conv10)
    
    h_conv11 = tf.nn.relu(conv2d(h_pool10 , w_conv11)+b_conv11)
    h_conv12 = tf.nn.relu(conv2d(h_conv11 , w_conv12)+b_conv12)              
    h_conv13= tf.nn.relu(conv2d(h_conv12 , w_conv13)+b_conv13)
    h_pool13 = max_pool_2x2(h_conv13)
                         
                    
                         
    
    
    
    
    
    
    h_pool5= max_pool_2x2(h_conv5)

    print conv2d(x_image , w_conv1)+b_conv1
    print h_conv1
    
    print h_conv2
    print h_conv3
    print h_conv4

    print b_conv2.get_shape()
    
    #print conv2d(h_pool1 , w_conv2).get_shape()


# In[8]:

print w_conv1.get_shape()


# In[9]:

pooling_col_size4=int(h_pool13.get_shape()[2])
pooling_row_size4=int(h_pool13.get_shape()[1])
#connect fully connected layer 
with tf.device('/gpu:2'):
    w_fc1=tf.get_variable("fc1",[pooling_col_size4*pooling_row_size4*out_ch13,fully_ch1] , initializer = tf.contrib.layers.xavier_initializer())
    b_fc1 = bias_variable([fully_ch1])

    h_pool5_flat =tf.reshape(h_pool13, [-1,pooling_col_size4*pooling_row_size4*out_ch13])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat , w_fc1)+ b_fc1)
with tf.device('/gpu:3'):
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    w_fc2 =tf.get_variable("fc2",[fully_ch1 , n_classes],initializer = tf.contrib.layers.xavier_initializer())
    b_fc2 = bias_variable([n_classes])

    y_conv=tf.add(tf.matmul(h_fc1_drop,w_fc2),b_fc2)


# In[10]:

#dirname = '/home/ncc/notebook/mammo/result/'

dirname='/mnt/Jupyter/seongjung_mnt/Eye/result/'

count=0
while(True):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
        break
    elif not os.path.isdir(dirname + str(count)):
        dirname=dirname+str(count)
        os.mkdir(dirname)
        break
    else:
        count+=1
print 'it is recorded at :'+str(count)


# In[ ]:

f=open(dirname+"/log.txt",'w')


# In[ ]:

with tf.device('/gpu:2'):
#sm_conv= tf.nn.softmax(y_conv)
    #cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    start_time = time.time()

    regular=0.01*(tf.reduce_sum(tf.square(y_conv)))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( y_conv, y_))
with tf.device('/gpu:3'):
    cost = cost+regular
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost) #1e-4
    with tf.name_scope("accuracy"):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_conv,1) ,tf.argmax(y_,1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction , "float")) 

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

for i in range(iterate):
    
    batch_xs , batch_ys = next_batch(batch_size, train_img , train_lab)
   # batch_val_xs  , batch_val_ys = next_batch(20 , val_img , val_lab)
    if i%100 ==0: # in here add to validation 
        try:
            train_accuracy = sess.run( accuracy , feed_dict={x:val_img , y_:val_lab , keep_prob: 1.0})        
            loss = sess.run(cost , feed_dict = {x:val_img , y_: val_lab , keep_prob: 1.0})

            #result = sess.run(sm_conv , feed_dict = {x:val_img , y_:batch_ys , keep_prob :1.0})
            print("step %d , training  accuracy %g" %(i,train_accuracy))
            print("step %d , loss : %g" %(i,loss))
            str_ = 'step :'+str(i)+'loss :'+str(loss) +'training accuracy :'+str(train_accuracy)+'\n'
            f.write(str_)
    
        except :
            list_acc=[]
            list_loss=[]
            n_divide=len(val_img)/batch_size
            for j in range(n_divide):
                
                # j*batch_size :(j+1)*batch_size
                train_accuracy,loss = sess.run([accuracy ,cost], feed_dict={x:val_img[ j*batch_size :(j+1)*batch_size] , y_:val_lab[ j*batch_size :(j+1)*batch_size ] , keep_prob: 1.0})        
                list_acc.append(float(train_accuracy))
                list_loss.append(float(loss))
            train_accuracy , loss=sess.run([accuracy,cost] , feed_dict={x:val_img[(j+1)*batch_size : ] , y_:val_lab[(j+1)*(batch_size) : ] , keep_prob : 1.0})
            #right above code have to modify
            
            list_acc.append(train_accuracy)
            list_loss.append(loss)
            list_acc=np.asarray(list_acc)
            list_loss= np.asarray(list_loss)
            
            train_accuracy=np.mean(list_acc)
            loss = np.mean(list_loss)
            
            #result = sess.run(sm_conv , feed_dict = {x:val_img , y_:batch_ys , keep_prob :1.0})
            print("step %d , training  accuracy %g" %(i,train_accuracy))
            print("step %d , loss : %g" %(i,loss))
            str_ = 'step :'+str(i)+'loss :'+str(loss) +'training accuracy :'+str(train_accuracy)+'\n'
            f.write(str_)
    
    
    sess.run(train_step ,feed_dict={x:batch_xs , y_:batch_ys , keep_prob : 0.7})
print("test accuracy %g" %sess.run(accuracy , 
                                   feed_dict={x:test_img , y_:test_lab, keep_prob : 1.0 }))

print("--- Training Time : %s ---" % (time.time() - start_time))
str_='test accuracy'+str(accuracy)
f.write(str_)
#추가 한 부분 
test_pred , test_acc = sess.run([y_conv , accuracy] , feed_dict={x:test_img , y_:test_lab,keep_prob : 1.0})
f.write(str(test_acc))

sess.close()


# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def file2Graph(file_path , save_path , graph_size):
    assert type(graph_size) == tuple 
    
    step=[]
    loss=[]
    acc=[]
    test_acc=0.0
    fo=open(file_path)
    lines=fo.readlines()
    for line in lines:
        list_splited_line=line.split()
        if 'step' in list_splited_line:
            print list_splited_line[2]
            step.append(float(list_splited_line[2]))
            
        if 'loss' in line:
            loss.append(float(list_splited_line[5]))
        if 'training accuracy' in line:
            acc.append(float(list_splited_line[9]))
        if 'test accuracy' in line:
            test_acc=float(list_splited_line[3])
            
    #need 4 4graph 
    fig1 = plt.figure(figsize =graph_size)
    save_name='x: step , y : loss'
    ax=fig1.add_subplot(1,1,1)
    plt.plot(step , loss)
    plt.savefig(save_path+save_name)
    
    fig2 = plt.figure(figsize =graph_size)
    ax1=fig2.add_subplot(1,1,1)
    save_name='x: step , y : acc'
    plt.plot(step , acc)
    plt.savefig(save_path+save_name)
    
    fig3 = plt.figure(figsize =graph_size)
    ax2=fig3.add_subplot(1,1,1)
    
    save_name='x: step , y(blue) : acc , y2(green) : loss'
    plt.plot(step , acc ,'b')
    plt.plot(step , loss ,'g')
    plt.savefig(save_path+save_name)
    
    


# In[ ]:

import Image
result_list=list(result_np)
lab_list=list(lab_np)
pred_list=list(pred_np)
#
#def savepic(save_path,extension,img_source,lab_source, pred_list , result_list=None, img_row , img_col,color_ch=1 ):

import Utility_
save_path ='/home/ubuntu/Desktop/delete_'
extension='.jpg'
img_source = test_img
print np.shape(img_source)
if 'numpy'  in str(type(test_img)):
    print 'numpy'
print type(img_source)
img=Utility_.savepic(save_path , extension , img_source , lab_list , pred_list , result_list, img_row=64 ,                 img_col=64 , color_ch =1)
img=img.reshape(img_row , img_col)
print np.shape(img)
plt.imshow(img)


# In[ ]:

sess.close()


# In[ ]:



