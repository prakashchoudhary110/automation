{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 27,
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
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.layers import Convolution2D"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.layers import MaxPooling2D"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.layers import Flatten"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.layers import Dense"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import Sequential"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
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
   "execution_count": 35,
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
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "#model.add(MaxPooling2D(pool_size=(2, 2)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
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
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(MaxPooling2D(pool_size=(2, 2)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
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
   "execution_count": 40,
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
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Flatten())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
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
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Dense(units=128, activation='relu'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
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
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.add(Dense(units=1, activation='sigmoid'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
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
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras_preprocessing.image import ImageDataGenerator"
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
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 41 images belonging to 3 classes.\n",
      "Found 6 images belonging to 3 classes.\n",
      "Epoch 1/5\n",
      "52/60 [=========================>....] - ETA: 18s - loss: -200663.3241 - accuracy: 0.0732"
     ]
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
    "        steps_per_epoch=60,\n",
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
   "execution_count": null,
   "metadata": {},
   "outputs": [],
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
