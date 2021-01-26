import numpy as np                  # Linear algebra
import pandas as pd                 # Data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt     # Plotting    

import torch                        # Tensors
import torch.nn as nn               # For model
import torch.nn.functional as F     # For loss function
from torch.utils.data import DataLoader, TensorDataset, random_split # For putting data into batches, splitting etc

def formatData(df):       # get all the data into a usable form
    # Clean data
    df = df.replace({'\$':'', ',': ''}, regex = True)
    # Turn data numerical
    df = df.replace({'no':0,'No':0,'z_No':0,'z_F':0,'yes':1,'Yes':1,'M':1}, regex = True)
    df = df.replace({'Private':0,'Commercial':1,'Highly Rural/ Rural':0,'Highly Urban/ Urban':1}, regex = True)
    df = df.replace({'z_High School':0,'<High School':1,'Bachelors':2,'Masters':3,'PhD':4}, regex = True)
    df = df.replace({'Student':1,'z_Blue Collar':2,'Clerical':3,'Home Maker':4,'Professional':5,'Lawyer':6,'Manager':7,'Doctor':8}, regex = True)
    df = df.replace({'Sports Car':1,'Pickup':2,'z_SUV':3,'Van':4,'Panel Truck':5,'Minivan':6}, regex = True)
    # Format datatype
    df['INCOME'] = pd.to_numeric(df['INCOME'],errors='coerce')
    df['HOME_VAL'] = pd.to_numeric(df['HOME_VAL'],errors='coerce')
    df['BLUEBOOK'] = pd.to_numeric(df['BLUEBOOK'],errors='coerce')
    df['OLDCLAIM'] = pd.to_numeric(df['OLDCLAIM'],errors='coerce')    
    df=df.astype(np.float32)
  
    # Replace blanks - could edit homeval
    df=df.fillna(df.mean())

    # Convert from pandas to numpy
    inputs=df[['KIDSDRIV','AGE','HOMEKIDS','YOJ','PARENT1','MSTATUS','SEX','EDUCATION','JOB','TRAVTIME','CAR_USE','TIF','CAR_TYPE','RED_CAR','CLM_FREQ','REVOKED','MVR_PTS','CAR_AGE','URBANICITY']].to_numpy()
    #, 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM' - not formatting properly
    targets=df[['TARGET_FLAG','TARGET_AMT']].to_numpy()
    #,'TARGET_AMT' - could be connected to flag there is no amount if there is no flag do one then the other
    
    # Convert to tensors
    inputs=torch.from_numpy(inputs)    
    targets=torch.from_numpy(targets)
    
    return inputs,targets

# Split into validation and training data
def splitdata(dataset):
    num_rows = 8161 # inputs.shape[0]
    val_percent = 0.10 # size of validation set
    val_size = int(num_rows * val_percent)
    train_size = num_rows - val_size
    train_ds, val_ds = random_split(dataset, [train_size , val_size ])
    return train_ds, val_ds

# Batch the data
def dataLoader(train_ds,val_ds):
    batch_size = 1000       # variable

    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size)
    return train_loader, val_loader

# Function to train the model parameters
def fit(epochs, model, train_loader, val_loader, opt_func):
    history = []
    x=[]
    # Repeat for given number of epochs
    for epoch in range(epochs):
        x.append(epoch)

        # Train phase
        for inputs, targets in train_loader:
            predict = model(inputs)                 # Generate predictions
            loss = loss_fn(predict, targets)        # Calculate loss
            loss.backward()                         # Compute gradients
            opt_func.step()                         # Update parameters using gradients
            opt_func.zero_grad()                    # Reset the gradients to zero
        # Validation phase
        for inputs, targets in val_loader:
            with torch.no_grad():
                predict = model(inputs)                 # Generate predictions
                result = loss_fn(predict, targets)      # Calculate loss
                history.append(result)                  # Save the loss
        # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item())) 
        
    return x,history

# Function to plot 
def plotting(x,y,xlabel,ylabel):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y, 'go', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    # Get data from CSV
    train_df = pd.read_csv('./auto-insurance-fall-2017/train_auto.csv',dtype={'Name': str,'Grade': float},na_values = ['NaN', '']) # Import the data from the CSV
    inputs,targets = formatData(train_df)                       # Getting the data in a form that can be used
    train_dataset = TensorDataset(inputs, targets)              # Combining the targets and input into a tensor dataset
    train_ds, val_ds = splitdata(train_dataset)                 # Splitting into training and validation
    train_loader, val_loader = dataLoader(train_ds, val_ds)     # Putting the data in batches

    epochs=100                                                  # Define training time
    lr=1e-6                                                     # set learing rate
    model = nn.Linear(inputs.shape[1], targets.shape[1])        # Define model
    loss_fn = F.mse_loss                                        # Loss function
    opt_func = torch.optim.SGD(model.parameters(), lr)          # Define optimizer
    
    x,history = fit(epochs, model, train_loader, val_loader, opt_func)

    # Plot the validation loss over the epochs
    plotting(x, history, 'Epochs', 'Loss')
    
    # Seeing accuracy on orignal data set
    predict =np.array(model(inputs).detach())
    for i in range(len(predict)): # This is not the way to get the predictions
        if predict[i,0]>0.5:
            predict[i,0] = 1
        else:
            predict[i,0] = 0

    # Calculating accuracy
    predict = torch.from_numpy(predict)
    accuracy = torch.tensor(torch.sum(predict == targets).item() / len(predict))
    print(accuracy)

    # Applying the model to the test data
    test_df = pd.read_csv('./auto-insurance-fall-2017/test_auto.csv',dtype={'Name': str,'Grade': float},na_values = ['NaN', ''])
    test_inputs,test_targets = formatData(test_df)

    # Generating predictions
    predict =np.array(model(test_inputs).detach())

    for i in range(len(predict)):   # Again this is not how to get the results of the model
        if predict[i,0]>0.5:        # Should do a softmax for the flag
            predict[i,0] = 1
        else:
            predict[i,0] = 0

    pd.DataFrame(predict).to_csv("results.csv")

