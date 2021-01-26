import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Get data from CSV
train = pd.read_csv('./auto-insurance-fall-2017/train_auto.csv',dtype={'Name': str,'Grade': float},na_values = ['NaN', ''])

# Clean data
train = train.replace({'\$':'', ',': ''}, regex = True)
# Turn data numerical
train = train.replace({'no':0,'No':0,'z_No':0,'z_F':0,'yes':1,'Yes':1,'M':1}, regex = True)
train = train.replace({'Private':0,'Commercial':1,'Highly Rural/ Rural':0,'Highly Urban/ Urban':1}, regex = True)
train = train.replace({'z_High School':0,'<High School':1,'Bachelors':2,'Masters':3,'PhD':4}, regex = True)
train = train.replace({'Student':1,'z_Blue Collar':2,'Clerical':3,'Home Maker':4,'Professional':5,'Lawyer':6,'Manager':7,'Doctor':8}, regex = True)
train = train.replace({'Sports Car':1,'Pickup':2,'z_SUV':3,'Van':4,'Panel Truck':5,'Minivan':6}, regex = True)
# Format datatype
train['INCOME'] = pd.to_numeric(train['INCOME'],errors='coerce')
train['HOME_VAL'] = pd.to_numeric(train['HOME_VAL'],errors='coerce')
train['BLUEBOOK'] = pd.to_numeric(train['BLUEBOOK'],errors='coerce')
train['OLDCLAIM'] = pd.to_numeric(train['OLDCLAIM'],errors='coerce')
train=train.astype(np.float32)

# Replace blanks - could edit homeval, zero is renting
train=train.fillna(train.mean())

# Convert from pandas to numpy
inputs=train[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','KIDSDRIV', 'AGE','HOMEKIDS','YOJ','PARENT1','MSTATUS','SEX','EDUCATION','JOB','TRAVTIME','CAR_USE','TIF','CAR_TYPE','RED_CAR','CLM_FREQ','REVOKED','MVR_PTS','CAR_AGE','URBANICITY']].to_numpy()
targets=train[['TARGET_FLAG','TARGET_AMT']].to_numpy()      #, - should be connected to flag there is no amount if there is no flag do one then the other

# Convert to tensors
inputs=torch.from_numpy(inputs)    
targets=torch.from_numpy(targets)

# Transpose data
input_dataset = inputs.t()
target_dataset = targets.t()

# Prameters weights and biases
A = torch.randn((1, input_dataset.shape[0]), requires_grad=True)
b = torch.randn(1, requires_grad=True)

# For saving the history
history = []
x=[]

# Define the prediction model
def model(input):
    return A.mm(input) + b

# Loss function
loss_fn = F.mse_loss                                        

# Chose and optimiser
opt = optim.Adam([A, b], lr=0.1)

no_epochs=100
# Main optimization loop
for epoch in range(no_epochs):
    opt.zero_grad()                             # Set the gradients to 0
    predict = model(input_dataset)              # Calculate predictions
    loss = loss_fn(predict, target_dataset)     # See how far off the prediction is
    loss.backward()                             # Compute the gradient
    opt.step()                                  # Update parameters

    # Print the progress
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{no_epochs}], Loss: {loss.item()}')

        #save progress
        x.append(epoch)
        history.append(loss)

# plot how the model improves
plt.plot(x, history, 'go', alpha=0.5)
plt.show()

#need to do a soft max 
_, preds = torch.max(predict, dim=1)
accuracy = torch.tensor(torch.sum(preds == targets).item() / len(preds))
