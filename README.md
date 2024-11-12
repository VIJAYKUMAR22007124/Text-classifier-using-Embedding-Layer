# Text-classifier-using-Embedding-Layer
## AIM
To create a classifier using specialized layers for text data such as Embedding and GlobalAveragePooling1D.

## PROBLEM STATEMENT AND DATASET

<img width="755" alt="{F5316EDD-9182-464D-8362-769E91BE081A}" src="https://github.com/user-attachments/assets/781c340a-bc26-4efa-873b-f43aa5e479c1">

<hr>
The objective of this project is to develop a text classification model that categorizes BBC news articles into relevant topics. Using a dataset of labeled news articles, we will preprocess the text data, tokenize it, and build a neural network model to classify the articles into predefined categories. The model's performance will be validated on unseen data to assess its accuracy. This solution aims to enhance automated content classification in news media.

## DESIGN STEPS

### STEP 1: 
Extract the contents of the zip file to access the BBC news dataset. Split this dataset into two parts: one for training and the other for validation.

### STEP 2:
Define a preprocessing function to clean the text data by converting all text to lowercase, removing stop words, and eliminating any punctuation.

### STEP 3:
Create a TextVectorizer layer to tokenize the text data and transform it into sequences, making it ready for model training.

### STEP 4:
Use TensorFlow's StringLookup layer to convert text labels into a numerical format that the model can understand.

### STEP 5:
Design a neural network model for multi-class classification. The architecture should include an embedding layer, a global average pooling layer, and dense layers.

### STEP 6:
Train the model for 30 epochs on the training dataset and validate its performance using the validation set.

### STEP 7:
Measure the model's accuracy and loss, then plot these metrics over the epochs to visualize and track performance.

## PROGRAM
### Name: B VIJAY KUMAR

### Register Number: 212222230173

### Import Libraries
```
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import zipfile

```

### Extract Data
```
with zipfile.ZipFile('/content/BBC News Train.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('extracted_data')
with open("extracted_data/BBC News Train.csv", 'r') as csvfile:
    print(f"First line (header) looks like this:\n\n{csvfile.readline()}")
    print(f"The second line (first data point) looks like this:\n\n{csvfile.readline()}")

```

### Define Global Variables
```
VOCAB_SIZE = 1000
EMBEDDING_DIM = 16
MAX_LENGTH = 120
TRAINING_SPLIT = 0.8

```

### Load and Display Data
```
data_dir = "/content/extracted_data/BBC News Train.csv"
data = np.loadtxt(data_dir, delimiter=',', skiprows=1, dtype='str', comments=None)
print(f"Shape of the data: {data.shape}")
print(f"{data[0]}\n{data[1]}")

```

### Data Summary
```
print(f"There are {len(data)} sentence-label pairs in the dataset.\n")
print(f"First sentence has {len((data[0,1]).split())} words.\n")
print(f"The first 5 labels are {data[:5,2]}")
```

###  Train-Validation Split Function
```
def train_val_datasets(data):
    # Define the training size (e.g., 80% of the total data)
    train_size = int(0.8 * len(data))
    
    # Slice the dataset to get texts and labels
    texts = data[:, 1]
    labels = data[:, 2]
    
    # Split the sentences and labels into train/validation sets
    train_texts = texts[:train_size]
    validation_texts = texts[train_size:]
    train_labels = labels[:train_size]
    validation_labels = labels[train_size:]
    
    # Create the train and validation datasets from the splits
    train_dataset = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_texts, validation_labels))
    
    
    return train_dataset, validation_dataset
```

### Create Train and Validation Datasets
```
train_dataset, validation_dataset = train_val_datasets(data)
print('Name:   B VIJAY KUMAR     Register Number:   212222230173    ')
print(f"There are {train_dataset.cardinality()} sentence-label pairs for training.\n")
print(f"There are {validation_dataset.cardinality()} sentence-label pairs for validation.\n")
```

### Text Standardization Function
```
def standardize_func(sentence):
    # List of stopwords
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here",  "hers", "herself", "him", "himself", "his", "how",  "i", "if", "in", "into", "is", "it", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she",  "should", "so", "some", "such", "than", "that",  "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",  "were", "what",  "when", "where", "which", "while", "who", "whom", "why", "why", "with", "would", "you",  "your", "yours", "yourself", "yourselves", "'m",  "'d", "'ll", "'re", "'ve", "'s", "'d"]
 
    # Sentence converted to lowercase-only
    sentence = tf.strings.lower(sentence)
    
    # Remove stopwords
    for word in stopwords:
        if word[0] == "'":
            sentence = tf.strings.regex_replace(sentence, rf"{word}\b", "")
        else:
            sentence = tf.strings.regex_replace(sentence, rf"\b{word}\b", "")
    
    # Remove punctuation
    sentence = tf.strings.regex_replace(sentence, r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']', "")
    
    return sentence
     
```

### Fit Text Vectorizer
```
def fit_vectorizer(train_sentences, standardize_func):
    
    # Instantiate the TextVectorization class, passing in the correct values for the parameters
    vectorizer = tf.keras.layers.TextVectorization(
        standardize=standardize_func,            # Custom standardization function
        max_tokens=VOCAB_SIZE,                   # Maximum vocabulary size
        output_sequence_length=MAX_LENGTH        # Truncate sequences to this length
    )
    
    # Adapt the vectorizer to the training sentences
    vectorizer.adapt(train_sentences)
    
    
    return vectorizer

text_only_dataset = train_dataset.map(lambda text, label: text)
vectorizer = fit_vectorizer(text_only_dataset, standardize_func)
vocab_size = vectorizer.vocabulary_size()
print('Name: B VIJAY KUMAR    Register Number:  212222230173     ')
print(f"Vocabulary contains {vocab_size} words\n")
```

### Fit Label Encoder

```
def fit_label_encoder(train_labels, validation_labels):
  
    # Concatenate the training and validation label datasets
    labels = train_labels.concatenate(validation_labels)
    
    # Instantiate the StringLookup layer without any OOV token
    label_encoder = tf.keras.layers.StringLookup(num_oov_indices=0)
    
    # Adapt the StringLookup layer on the combined labels dataset
    label_encoder.adapt(labels)
    
    
    return label_encoder

train_labels_only = train_dataset.map(lambda text, label: label)
validation_labels_only = validation_dataset.map(lambda text, label: label)

label_encoder = fit_label_encoder(train_labels_only,validation_labels_only)
print('Name:  B VIJAY KUMAR      Register Number:   212222230173    ')
print(f'Unique labels: {label_encoder.get_vocabulary()}')

```

### Preprocess Dataset

```
def preprocess_dataset(dataset, text_vectorizer, label_encoder):
    """Apply the preprocessing to a dataset

    Args:
        dataset (tf.data.Dataset): dataset to preprocess
        text_vectorizer (tf.keras.layers.TextVectorization ): text vectorizer
        label_encoder (tf.keras.layers.StringLookup): label encoder

    Returns:
        tf.data.Dataset: transformed dataset
    """

      ### START CODE HERE ###

    # Apply text vectorization and label encoding
    dataset = dataset.map(lambda text, label: (text_vectorizer(text), label_encoder(label)))

    # Set the batch size to 32
    dataset = dataset.batch(32)

    ### END CODE HERE ###

    return dataset

train_proc_dataset = preprocess_dataset(train_dataset, vectorizer, label_encoder)
validation_proc_dataset = preprocess_dataset(validation_dataset, vectorizer, label_encoder)

train_batch = next(train_proc_dataset.as_numpy_iterator())
validation_batch = next(validation_proc_dataset.as_numpy_iterator())
print('Name:  B VIJAY KUMAR      Register Number:   212222230173    ')
print(f"Shape of the train batch: {train_batch[0].shape}")
print(f"Shape of the validation batch: {validation_batch[0].shape}")


```

### Define and Compile Model

```

# GRADED FUNCTION: create_model
def create_model():
    """
    Creates a text classifier model
    Returns:
      tf.keras Model: the text classifier model
    """

    # Define your model
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(MAX_LENGTH,)),
        tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    # Compile model. Set an appropriate loss, optimizer and metrics
    model.compile(
        loss='sparse_categorical_crossentropy',  # or 'categorical_crossentropy' if labels are one-hot encoded
        optimizer='adam',
        metrics=['accuracy'] 
    ) 



    return model

model = create_model()

example_batch = train_proc_dataset.take(1)

try:
	model.evaluate(example_batch, verbose=False)
except:
	print("Your model is not compatible with the dataset you defined earlier. Check that the loss function and last layer are compatible with one another.")
else:
	predictions = model.predict(example_batch, verbose=False)
	print(f"predictions have shape: {predictions.shape}")



```

### Train Model

```
history = model.fit(train_proc_dataset, epochs=30, validation_data=validation_proc_dataset)
```

### Plot Training Graphs
```

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, f'val_{metric}'])
    plt.show()
print('Name:    B VIJAY KUMAR    Register Number:    212222230173   ')
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

```

## OUTPUT
### Loss, Accuracy Vs Iteration Plot

<img width="225" alt="{89FD8D6F-E0FD-4232-B10D-1DEDD24C9284}" src="https://github.com/user-attachments/assets/1d35809a-f337-41bd-896f-c02b486bfe88">


## RESULT


The text classifier was successfully implemented using layers like Embedding, GlobalAveragePooling1D, and Dense. The model was trained and validated on the BBC news dataset, achieving a reliable classification performance. The evaluation metrics confirmed the model's capability to accurately categorize news articles into their respective topics.
