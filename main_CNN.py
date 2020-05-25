{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "#!/usr/bin/env python\n",
    "\n",
    "# coding: utf-8"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.layers import Convolution2D"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.layers import MaxPooling2D"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.layers import Flatten"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.layers import Dense"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import Sequential"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Convolution2D(filters=32, \n",
    "\n",
    "                        kernel_size=(3,3), \n",
    "\n",
    "                        activation='relu',\n",
    "\n",
    "                   input_shape=(64, 64, 3)\n",
    "\n",
    "                       ))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_2\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d_4 (Conv2D)            (None, 62, 62, 32)        896       \n",
      "=================================================================\n",
      "Total params: 896\n",
      "Trainable params: 896\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "Input 0 is incompatible with layer max_pooling2d_3: expected ndim=4, found ndim=2",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-51-1308ef673ad4>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0madd\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mMaxPooling2D\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mpool_size\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m2\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m2\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\keras\\engine\\sequential.py\u001b[0m in \u001b[0;36madd\u001b[1;34m(self, layer)\u001b[0m\n\u001b[0;32m    180\u001b[0m                 \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0minputs\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnetwork\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_source_inputs\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0moutputs\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    181\u001b[0m         \u001b[1;32melif\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0moutputs\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 182\u001b[1;33m             \u001b[0moutput_tensor\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mlayer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0moutputs\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    183\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0misinstance\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0moutput_tensor\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mlist\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    184\u001b[0m                 raise TypeError('All layers in a Sequential model '\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\keras\\backend\\tensorflow_backend.py\u001b[0m in \u001b[0;36msymbolic_fn_wrapper\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m     73\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0m_SYMBOLIC_SCOPE\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mvalue\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     74\u001b[0m             \u001b[1;32mwith\u001b[0m \u001b[0mget_graph\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mas_default\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 75\u001b[1;33m                 \u001b[1;32mreturn\u001b[0m \u001b[0mfunc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     76\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     77\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0mfunc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\keras\\engine\\base_layer.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, inputs, **kwargs)\u001b[0m\n\u001b[0;32m    444\u001b[0m                 \u001b[1;31m# Raise exceptions in case the input is not compatible\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    445\u001b[0m                 \u001b[1;31m# with the input_spec specified in the layer constructor.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 446\u001b[1;33m                 \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0massert_input_compatibility\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0minputs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    447\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    448\u001b[0m                 \u001b[1;31m# Collect input shapes to build layer.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\keras\\engine\\base_layer.py\u001b[0m in \u001b[0;36massert_input_compatibility\u001b[1;34m(self, inputs)\u001b[0m\n\u001b[0;32m    340\u001b[0m                                      \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mname\u001b[0m \u001b[1;33m+\u001b[0m \u001b[1;34m': expected ndim='\u001b[0m \u001b[1;33m+\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    341\u001b[0m                                      \u001b[0mstr\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mspec\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mndim\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;33m+\u001b[0m \u001b[1;34m', found ndim='\u001b[0m \u001b[1;33m+\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 342\u001b[1;33m                                      str(K.ndim(x)))\n\u001b[0m\u001b[0;32m    343\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mspec\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mmax_ndim\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    344\u001b[0m                 \u001b[0mndim\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mK\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mndim\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mValueError\u001b[0m: Input 0 is incompatible with layer max_pooling2d_3: expected ndim=4, found ndim=2"
     ]
    }
   ],
   "source": [
    "model.add(MaxPooling2D(pool_size=(2, 2)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Convolution2D(filters=32, \n",
    "\n",
    "                        kernel_size=(3,3), \n",
    "\n",
    "                        activation='relu',\n",
    "\n",
    "                       ))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(MaxPooling2D(pool_size=(2, 2)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Convolution2D(filters=32, \n",
    "\n",
    "                        kernel_size=(3,3), \n",
    "\n",
    "                        activation='relu',\n",
    "\n",
    "                       ))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_2\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d_4 (Conv2D)            (None, 62, 62, 32)        896       \n",
      "_________________________________________________________________\n",
      "conv2d_5 (Conv2D)            (None, 60, 60, 32)        9248      \n",
      "_________________________________________________________________\n",
      "max_pooling2d_2 (MaxPooling2 (None, 30, 30, 32)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_6 (Conv2D)            (None, 28, 28, 32)        9248      \n",
      "=================================================================\n",
      "Total params: 19,392\n",
      "Trainable params: 19,392\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Flatten())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_2\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d_4 (Conv2D)            (None, 62, 62, 32)        896       \n",
      "_________________________________________________________________\n",
      "conv2d_5 (Conv2D)            (None, 60, 60, 32)        9248      \n",
      "_________________________________________________________________\n",
      "max_pooling2d_2 (MaxPooling2 (None, 30, 30, 32)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_6 (Conv2D)            (None, 28, 28, 32)        9248      \n",
      "_________________________________________________________________\n",
      "flatten_2 (Flatten)          (None, 25088)             0         \n",
      "=================================================================\n",
      "Total params: 19,392\n",
      "Trainable params: 19,392\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Dense(units=128, activation='relu'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_2\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d_4 (Conv2D)            (None, 62, 62, 32)        896       \n",
      "_________________________________________________________________\n",
      "conv2d_5 (Conv2D)            (None, 60, 60, 32)        9248      \n",
      "_________________________________________________________________\n",
      "max_pooling2d_2 (MaxPooling2 (None, 30, 30, 32)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_6 (Conv2D)            (None, 28, 28, 32)        9248      \n",
      "_________________________________________________________________\n",
      "flatten_2 (Flatten)          (None, 25088)             0         \n",
      "_________________________________________________________________\n",
      "dense_3 (Dense)              (None, 128)               3211392   \n",
      "=================================================================\n",
      "Total params: 3,230,784\n",
      "Trainable params: 3,230,784\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Dense(units=1, activation='sigmoid'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_2\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d_4 (Conv2D)            (None, 62, 62, 32)        896       \n",
      "_________________________________________________________________\n",
      "conv2d_5 (Conv2D)            (None, 60, 60, 32)        9248      \n",
      "_________________________________________________________________\n",
      "max_pooling2d_2 (MaxPooling2 (None, 30, 30, 32)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_6 (Conv2D)            (None, 28, 28, 32)        9248      \n",
      "_________________________________________________________________\n",
      "flatten_2 (Flatten)          (None, 25088)             0         \n",
      "_________________________________________________________________\n",
      "dense_3 (Dense)              (None, 128)               3211392   \n",
      "_________________________________________________________________\n",
      "dense_4 (Dense)              (None, 1)                 129       \n",
      "=================================================================\n",
      "Total params: 3,230,913\n",
      "Trainable params: 3,230,913\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras_preprocessing.image import ImageDataGenerator"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5\n",
      "\n"
     ]
    }
   ],
   "source": [
    "f=open(\"val.txt\",\"r\")\n",
    "\n",
    "value=f.readline()\n",
    "\n",
    "if value == \"\":\n",
    "\n",
    "    value='3'\n",
    "\n",
    "print(value)\n",
    "\n",
    "f.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 41 images belonging to 3 classes.\n",
      "Found 6 images belonging to 3 classes.\n",
      "Epoch 1/5\n",
      "49/50 [============================>.] - ETA: 3s - loss: -157601.9112 - accuracy: 0.0738"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\sai\\Anaconda3\\lib\\site-packages\\keras\\utils\\data_utils.py:616: UserWarning: The input 0 could not be retrieved. It could be because a worker has died.\n",
      "  UserWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "50/50 [==============================] - 402s 8s/step - loss: -184005.7107 - accuracy: 0.0741 - val_loss: 282442.7188 - val_accuracy: 0.3333\n",
      "Epoch 2/5\n",
      "50/50 [==============================] - 285s 6s/step - loss: -33117791.3512 - accuracy: 0.0732 - val_loss: 26033120.0000 - val_accuracy: 0.3333\n",
      "Epoch 3/5\n",
      "50/50 [==============================] - 266s 5s/step - loss: -658532561.2564 - accuracy: 0.0732 - val_loss: 359832544.0000 - val_accuracy: 0.3333\n",
      "Epoch 4/5\n",
      "50/50 [==============================] - 303s 6s/step - loss: -5093320717.9153 - accuracy: 0.0732 - val_loss: 2208063744.0000 - val_accuracy: 0.3333\n",
      "Epoch 5/5\n",
      "50/50 [==============================] - 267s 5s/step - loss: -24472720296.8694 - accuracy: 0.0732 - val_loss: 8845852672.0000 - val_accuracy: 0.3333\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.callbacks.History at 0x206aa296f08>"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_datagen = ImageDataGenerator(\n",
    "\n",
    "        rescale=1./255,\n",
    "\n",
    "        shear_range=0.2,\n",
    "\n",
    "        zoom_range=0.2,\n",
    "\n",
    "        horizontal_flip=True)\n",
    "\n",
    "test_datagen = ImageDataGenerator(rescale=1./255)\n",
    "\n",
    "training_set = train_datagen.flow_from_directory(\n",
    "\n",
    "        'family/family/train/',\n",
    "\n",
    "        target_size=(64, 64),\n",
    "\n",
    "        batch_size=32,\n",
    "\n",
    "        class_mode='binary')\n",
    "\n",
    "test_set = test_datagen.flow_from_directory(\n",
    "\n",
    "        'family/family/validation/',\n",
    "\n",
    "        target_size=(64, 64),\n",
    "\n",
    "        batch_size=32,\n",
    "\n",
    "        class_mode='binary')\n",
    "\n",
    "model.fit(\n",
    "\n",
    "        training_set,\n",
    "\n",
    "        steps_per_epoch=50,\n",
    "\n",
    "        epochs=int(value),\n",
    "\n",
    "        validation_data=test_set,\n",
    "\n",
    "        validation_steps=800)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.07414634\n",
      "\n",
      "new\n"
     ]
    }
   ],
   "source": [
    "history=model.history\n",
    "\n",
    "acc=history.history['accuracy']\n",
    "\n",
    "max_acc=max(acc)\n",
    "\n",
    "print(max_acc)\n",
    "\n",
    "reader=open(\"accuracy.txt\",'r')\n",
    "\n",
    "old=reader.read()\n",
    "\n",
    "print(old)\n",
    "\n",
    "if old<str(max_acc):\n",
    "\n",
    "    print(\"new\")\n",
    "\n",
    "    writer=open(\"accuracy.txt\",'w+')\n",
    "\n",
    "    writer.write(str(acc))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    " "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
