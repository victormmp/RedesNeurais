#!/usr/bin/env python
# coding: utf-8

# # Regularização

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
import click


# ### Criação das Classes

# In[2]:


samples = 100
features = 2

class_1 = np.array(np.random.normal(2, 3, [samples, features]))
class_2 = np.array(np.random.normal(6, 3, [samples, features]))

y = np.concatenate(([0 for _ in range(100)], [1 for _ in range(100)]))

plt.scatter(class_1[:,0], class_1[:,1])
plt.scatter(class_2[:,0], class_2[:,1])


# ### Definição da Função de Treinamento

# In[3]:


def train(x, y, eta=0.01, epochs=20):
    
    # Add the bias parameter
    x_aug = np.column_stack((np.ones(x.shape[0]), x))
    
    # Initialize Weights
    w = np.random.normal(-0.5, 0.5, (x_aug.shape[1]))
    errors = []
#     df = pd.DataFrame(np.column_stack((x_aug, y)))


    # Start training
    for epoch in range(epochs):
        click.echo(f'Epoch {epoch + 1} of {epochs}')

        with click.progressbar(random.sample([num for num in range(x_aug.shape[0])], x_aug.shape[0])) as indexes:
            for i in indexes:
                y_hat = np.dot(x_aug[i, :], w)
                error = y[i] - y_hat
                w += eta * error * x_aug[i,:]
                errors.append(error)
        click.echo(f'  -> Error: ')

    return w, error
    


# In[4]:


x = np.concatenate([class_1, class_2])
train_index = np.random.randint(0, x.shape[0], int(0.7*x.shape[0]))

x_train = x[train_index]
x_test = x[~train_index]

y_train = y[train_index]
y_test = y[~train_index]


# In[5]:


train(x_train, y_train)

