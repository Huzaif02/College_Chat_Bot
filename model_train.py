# import numpy as np
# import pickle
# import tensorflow as tf
# from tensorflow.keras import layers, models, regularizers
# import json
# from voc import voc  # Ensure Voc class is imported from voc.py

# # Function to split the dataset
# # def splitDataset(data):
# #     if not data.questions:
# #         raise ValueError("No questions found in the dataset.")
    
# #     x_train = [data.getQuestionInNum(x) for x in data.questions]
# #     y_train = [data.getTag(data.questions[x]) for x in data.questions]
# #     return np.array(x_train), np.array(y_train)

# # # Load preprocessed data
# # print("Loading preprocessed data...")
# # with open("mydata.pickle", "rb") as f:
# #     data = pickle.load(f)

# def splitDataset(data):
#     x_train=[ data.getQuestionInNum(x) for x in data.questions]
#     y_train=[data.getTag(data.questions[x]) for x in data.questions]
#     return x_train,y_train
# with open("intents1.json") as file:
#     raw_data = json.load(file)

# data=voc()

# # Ensure that data is an instance of the Voc class
# if not isinstance(data, voc):
#     raise ValueError("Loaded data is not an instance of the Voc class")

# # Check the contents of data.questions
# print(f"Number of questions: {len(data.questions)}")
# if len(data.questions) > 0:
#     print(f"Sample question: {list(data.questions.keys())[0]}")
# else:
#     raise ValueError("The dataset contains no questions.")

# # Split the dataset
# print("Splitting dataset...")
# x_train, y_train = splitDataset(data)
# print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

# # Ensure x_train and y_train are not empty
# if x_train.size == 0 or y_train.size == 0:
#     raise ValueError("Training data is empty after processing.")

# # Build the model
# print("Building model...")
# model = models.Sequential()
# model.add(layers.Dense(128, input_shape=(x_train.shape[1],), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(y_train.shape[1], activation='softmax'))

# # Compile the model
# print("Compiling model...")
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# print("Training model...")
# model.fit(x_train, y_train, epochs=10, batch_size=8, verbose=1)

# # Save the model
# print("Saving model...")
# model.save('mymodel.h5')
# print("Model saved successfully.")




import numpy
import json
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import pickle
from voc import voc  
def splitDataset(data):
    x_train=[ data.getQuestionInNum(x) for x in data.questions]
    y_train=[data.getTag(data.questions[x]) for x in data.questions]
    return x_train,y_train
with open("intents1.json") as file:
    raw_data = json.load(file)

data=voc()

for intent in raw_data["intents"]:
    tag=intent["tag"]
    data.addTags(tag)

    for question in intent["patterns"]: 
        ques=question.lower()
        data.addQuestion(ques,tag)


x_train,y_train=splitDataset(data)
x_train=numpy.array(x_train)
y_train=numpy.array(y_train)
#normalize
#x_train=x_train/255
#reshape ytrain
'''
y_train = y_train.reshape((len(y_train), 1))

encoder = OneHotEncoder(sparse=False)
y_train=encoder.fit_transform(y_train)
'''


#intialising the ANN
model = models.Sequential()

# adding first layer
model.add(layers.Dense(units = 12, input_dim = len(x_train[0])))
model.add(layers.Activation('relu'))
#adding 2nd hidden layer
model.add(layers.Dense(units = 8))
model.add(layers.Activation('relu'))
#adding output layer
model.add(layers.Dense(units = 44))
model.add(layers.Activation('softmax'))

# Compiling the ANN
model.compile(optimizer = 'adam', loss ='categorical_crossentropy', metrics = ['accuracy'])
# Fitting the ANN model to training set
model.fit(x_train, y_train, batch_size = 10, epochs = 100)

model.save('mymodel.h5')



#removing questions from data as its not needed it will be entered by user
#we need other info to decode prediction to text so save it inpickle
data.questions={}

# save answers from json to pickle
for intent in raw_data["intents"]:
    tag=intent["tag"]
    response=[]
    for resp in intent["responses"]: 
        response.append(resp)
    data.addResponse(tag,response)
    
    
with open('mydata.pickle', 'wb') as handle:
    pickle.dump(data, handle)

 

# predecting the test set Results
x_test=numpy.array(x_train[0])
img = numpy.expand_dims(x_test, axis = 0)
y_pred = model.predict(img)
p=numpy.argmax(y_pred, axis=1)

