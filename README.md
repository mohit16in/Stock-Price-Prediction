# Stock Market Prediction using LSTM
   

The project revolves around the capability of LSTM to make predictions. The model tries to predict the next 30 days' closing price of the Stock Market.
The main aim of this project is to increase the accuracy of the prediction model by tweaking several **hyper-parameters**. The number of **LSTM** layers used would be fixed (50 units) but the parameters that are being changed are:- **BATCH_SIZE** and, **EPOCHS**.   
___
# Table of Contents
- [Abstract](https://github.com/mohit16in/Stock-Price-Prediction#abstract)
- [Background Information](https://github.com/mohit16in/Stock-Price-Prediction#background-information)
   - [What is Machine Learning?](https://github.com/mohit16in/Stock-Price-Prediction#what-is-machine-learning)
   - [How is it Different from Deep Learning?](https://github.com/mohit16in/Stock-Price-Prediction#how-is-it-different-from-deep-learning)
   - [AI vs ML vs DL](https://github.com/mohit16in/Stock-Price-Prediction#artificial-intelligence-vs-machine-learning-vs-deep-learning)
   - [Neural Networks](https://github.com/mohit16in/Stock-Price-Prediction#neural-networks)
   - [Simplest Neural Network](https://github.com/mohit16in/Stock-Price-Prediction#simplest-neural-network)
   - [Deep Neural Network](https://github.com/mohit16in/Stock-Price-Prediction#deep-neural-network)
   - [Recurrent Neural Network](https://github.com/mohit16in/Stock-Price-Prediction#recurrent-nueral-network)
   - [Long Short Term Memory](https://github.com/mohit16in/Stock-Price-Prediction#long-short-term-memory)
- [Project Disposition](https://github.com/mohit16in/Stock-Price-Prediction#project-disposition)
   - [Description](https://github.com/mohit16in/Stock-Price-Prediction#description)
   - [Plan Of Attack](https://github.com/mohit16in/Stock-Price-Prediction#plan-of-attack)
   - [Data Acquisition](https://github.com/mohit16in/Stock-Price-Prediction#data-acquisition)
   - [Data Preprocessing](https://github.com/mohit16in/Stock-Price-Prediction#data-preprocessing)
      - [Data Cleaning](https://github.com/mohit16in/Stock-Price-Prediction#1-data-cleaning)
      - [Data Extraction](https://github.com/mohit16in/Stock-Price-Prediction#2-data-extraction)
      - [Feature Scaling](https://github.com/mohit16in/Stock-Price-Prediction#3-feature-scaling)
   - [Structuring Data](https://github.com/mohit16in/Stock-Price-Prediction#structuring-data)
   - [Creating the model](https://github.com/mohit16in/Stock-Price-Prediction#creating-the-model)
   - [Training the model](https://github.com/mohit16in/Stock-Price-Prediction#training-the-model)
   - [Prediction](https://github.com/mohit16in/Stock-Price-Prediction#prediction)
   - [Plotting the Chart](https://github.com/mohit16in/Stock-Price-Prediction#plotting-the-chart)
- [Tweaking the Hyper-parameters](https://github.com/mohit16in/Stock-Price-Prediction#tweaking-the-hyper-parameters)

## Abstract    
**LSTM** or **Long Short-Term Memory** is an improvement over traditional **RNN** or **Recurrent Neural Network** in the sense that it can effectively “remember” **long sequence of events in the past**. Just like humans can derive information from the previous context and can chart his future actions, **RNN** and **LSTM** tends to imitate the same. The difference between **RNN** and **LSTM** is that **RNN** is used in places where the **retention of memory is short**, whereas **LSTM** is capable of connecting events that happened way earlier and the events that followed them.   
   
Hence, it (LSTM) is one of the **best** choices when it comes to analyzing and predicting **temporal dependent** phenomena which spans over a **long period** of time or multiple instances in the past.   
   
**LSTM** is capable of performing three main operations: **Forget**, **Input** and **Output**. These operations are made possible with the help of trained neural network layers like the **tanh layer** and the **sigmoid layer**. These layers decide whether an information needs to be **forgotten**, **updated** and what values need to be given as **output**. **LSTM** learns which parameter to learn, what to forget and what to be updated during the **training process**. That is the reason why **LSTM** requires a **huge amount of dataset** during its training and validation process for a **better result**. The **sigmoid layer** decides the flow of information. Whether it needs to allow all of the information or some of it or nothing.   
Multiple gates perform the roles of **Forget**, **Input** and **Output**. These gates perform the respective operation on the **cell state** which carries information from the past. The **forget gate layer** tells the **LSTM** cell whether to keep the past information or completely throw away. The **input gate** determines what new information should be added to the cell state. The **output gate** finally gives the output which is a filtered version of the cell state.

## Background Information
### What is Machine Learning?
Machine learning is a branch of **artificial intelligence (AI)** and **computer science** which focuses on the use of data and algorithms to imitate the way that humans learn, gradually **improving its accuracy**.
### How is it Different from Deep Learning?
- **Machine Learning** is the superset. **Deep Learning** is the subset.
- ML is the ability of computers to learn and act with less human intervention.
- DL is all about mimicking the thinking capability of the human brain and arriving at a conclusion just like a human does after analyzing and thinking about it for a while.
### Artificial Intelligence vs Machine Learning vs Deep Learning   
![image](https://user-images.githubusercontent.com/55954313/133870891-0423a591-7518-45e7-8bfc-a125b82fa160.png)
### Neural Networks
**Neural networks**, also known as **Artificial Neural Networks (ANNs)** are a subset of **Machine Learning** and are at the **heart of Deep Learning algorithms**. Their name and structure are inspired by the human brain, mimicking the way that biological neurons signal to one another.   
   
It comprises of several **nodes** which represent **neurons**. A collection of nodes creates a **node layer**. These node layers play specific roles. Some acts as **input layer**, some **hidden layers** and some acts as **output layer**. Each **node**, or **artificial neuron**, connects to another and has an **associated weight and threshold**. It is by tweaking these weights and thresholds, that the network is able to learn progressively.   
   
There are different types of neural nets : **Convoluted Neural Networks**, **Recurrent Neural Networks**, etc.
### Simplest Neural Network
![image](https://user-images.githubusercontent.com/55954313/133871109-a721309a-9d14-435f-8d32-167c6b15c0d8.png)   
This is called a **perceptron**. It has only one **input layer**, one **hidden layer** comprising of only one **node** and one **output layer**.   
#### Source: https://www.ibm.com/in-en/cloud/learn/neural-networks
### Deep Neural Network
![image](https://user-images.githubusercontent.com/55954313/133872209-40ad4e68-71de-47d4-984d-499599da59f1.png)   
- Contains multiple hidden layers
- Since the depth of hidden layers is deep, hence the name.
#### Source: https://www.ibm.com/in-en/cloud/learn/neural-networks
### Recurrent Nueral Network
![image](https://user-images.githubusercontent.com/55954313/133872292-e603b584-8ad2-43d1-8bdf-863d66e4c823.png)   
#### Source: https://colah.github.io/posts/2015-08-Understanding-LSTMs
- Uses sequential data or time series
- Used in solving temporal dependent problems: Natural language processing, image captioning, speech recognition, voice search, translation, etc.
- Inefficient when it comes to prediction where the outcome is long term temporal dependent.
- The reason being Exploding and Vanishing gradient.
- LSTM overcomes the drawback
### Long Short Term Memory
![image](https://user-images.githubusercontent.com/55954313/134108241-e04cf3cb-e1ef-4aa0-b594-e147ea11d5cb.png)
#### Source: https://colah.github.io/posts/2015-08-Understanding-LSTMs
- Contains a cell state that carries information from one state to another
- Gates manipulate the info in the cell state. Three main gates used to do this are: **Forget gate**, **Input gate**, **Output gate**.   
   
![image](https://user-images.githubusercontent.com/55954313/134108503-8be7d9d1-ddd3-4dc9-863b-09d48d60162f.png)
- **Forget Gate**: Responsible for removing useless information. 
- **Input Gate**: Responsible for adding new information.
- **Output Gate**: Responsible for filtering out the information.
___
# Project Disposition
## Description
- An attempt to predict the **Stock Market Price** using **LSTM** and analyze its accuracy by tweaking its **hyper-parameters**.
- For this purpose, **TSLA stock** has been used for training the model.
- The **line charts** of the model will be plotted, and its accuracy will be observed.
## Plan Of Attack
- Data Acquisition
- Data Preprocessing
- Structuring Data
- Creating the model
- Training the model
- Prediction
- Plotting chart and accuracy analysis
## Data Acquisition
![data collection tsla](https://github.com/mohit16in/Stock-Price-Prediction/assets/75472403/5a8337d4-ed39-4c55-8699-13c07eb8d9e0)

- [Yahoo finance](https://finance.yahoo.com/) maintains past data of hundreds of Stock Markets. One can easily download the data in the form of CSV files and use it for training.
- Past 4 years data is downloaded for TSLA stock.
  
![data head](https://github.com/mohit16in/Stock-Price-Prediction/assets/75472403/911d2256-cf2d-4440-baca-a84f8a2b725f)

## Data Preprocessing
### 1. Data Cleaning
One of the many important stages in creating an efficient model. Data contains **discrepancies**. If not removed, they might cause hindrance in producing accurate results. **Null** values are the most common.   

    
   
### 2. Data Extraction
Not all the data that we downloaded are necessary for training purposes. Fields like **Date** are unnecessary, hence we don't need them. In this project, I have chosen **Close** (which stores the **closing price of the market**) as the deciding variable that predicts the outcome. Remember that one can choose **multiple** deciding factors as well. The reason I have chosen the **Close** field is because, certainly, the **Stock Market price** depends on the previous days' closing prices. If I were to buy a Stock, I would definitely look at the closing price among other factors. Hence training the model on closing price seems feasible. But one can always opt for multiple deciding factors like considering both **opening** and **closing** prices together.   

Now the extracted data is stored in a 2D array for further processing.   
   
![close ](https://github.com/mohit16in/Stock-Price-Prediction/assets/75472403/72517ce5-e34b-484b-b120-bfb82842a651)

### 3. Feature Scaling
Feature scaling is the most important part of Data preprocessing. It helps in standardizing/normalizing the given dataset so that one entity is not given more importance than the other solely based on its quantity. Consider this as bringing all the scattered data within the same level for easier analysis. I have used MinMaxScaler() to scale all the values between 0 and 1.   

   
## Structuring Data
- The LSTM takes a series as an input. The length of the series is equivalent to the number of previous days. 
- X_train will be the input for the training set
- y_train is the output for the training set   
A similar data structure is created for preparing the prediction dataset.

Structuring the data for training purposes. The model will use the previous 30 days as the deciding factor and will predict the closing price for the next 30 days.

## Creating the model
![model](https://github.com/mohit16in/Stock-Price-Prediction/assets/75472403/c424fdc5-39b3-4940-9aee-ed386324ded6)
   

## Training the model
This is the stage where we play with the hyper-parameters to bring about changes in how the machine learns. The parameters that will be tweaked are: **BATCH_SIZE** and **EPOCHS** for epochs.

![hyperparameters](https://github.com/mohit16in/Stock-Price-Prediction/assets/75472403/cd65148a-b94f-42b7-a90f-4a4e0536e4db)
   

## Prediction
The model has been trained. Now it's time to predict the next day's price. For this purpose, a test dataset is created which contains past days' closing prices. This dataset is structured the same way as done in the Data Structuring step. Then these values are fed one by one to the LSTM input layer and the output is collected and stored in a 2D array for chart plotting and accuracy analysis.   
   

## Plotting the Chart
Based on the output we gathered, a line chart is plotted. The chart contains both the predicted and original price for accuracy analysis. From this chart, we can identify how the model performed.

![final outcome](https://github.com/mohit16in/Stock-Price-Prediction/assets/75472403/05ab3c3c-2141-4832-8f7d-087678171c67)

# Tweaking the Hyper-parameters
Tweaking hyperparameters brings about a change in how the model learns and analyzes the given data. Hyperparameters considered in this research project are Batch Size and number of Epochs.  

![hyperparameters](https://github.com/mohit16in/Stock-Price-Prediction/assets/75472403/8fea8cf8-bcd8-4879-9600-f3d44a194773)

![error](https://github.com/mohit16in/Stock-Price-Prediction/assets/75472403/70d09b76-986a-4fb9-a855-4ea59f2598f8)
