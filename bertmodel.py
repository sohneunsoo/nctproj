from eng_preprocessing import sentence_to_corpus , sentence_preprocessing
import numpy as np 
import pandas as pd 
import re , os
from transformers import *
import tensorflow as tf
from collections import Counter

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
MAX_LEN = 70

class TFBertClassifier_ENG(tf.keras.Model):
    def __init__(self, model_name):
        super(TFBertClassifier_ENG, self).__init__()

        self.bert = TFBertModel.from_pretrained(model_name)
        self.maxpooling = tf.keras.layers.GlobalMaxPool1D()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.d2= tf.keras.layers.Dense(32, activation = 'relu')
        self.classifier = tf.keras.layers.Dense(2, activation = 'sigmoid',
                                                 name="classifier")
        
        self.d1.trainable = True

    def get_config(self):
        
        config = super().get_config().copy()
        config.update({
            'bert':self.bert,
            'maxpooling':self.maxpooling,
            'dense1' : self.d1,
            'dropout':self.dropout,
            'dense2' : self.d2,
            'classifier':self.classifier,
        })
        return config
        
    def call(self, inputs, training=False):
        embeddings = self.bert(inputs)[0]
        pooled_output = self.maxpooling(embeddings)
        pooled_output = self.d1(pooled_output)
        pooled_output = self.dropout(pooled_output, training=training)
        pooled_output = self.d2(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def load_model_eng(checkpoint_path):
    # 모델 객체 생성
    model = TFBertClassifier_ENG(model_name='bert-base-cased')
    model.load_weights(checkpoint_path)
    return model

def predict(sentence, model):
    global MAX_LEN
    pass