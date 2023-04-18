# Sentiment-analysis-of-Yelp-Reviews

## Overivew

#### NLP application:

- Text classification and sentiment analysis: determining the topic and sentiment of a given text, for example, categorizing customer reviews as positive or negative
  - useful for analyzing customer feedback, social media posts, and reviews
  - performed at different levels, such as document level (determining the overall sentiment of a document), sentence level (determining the sentiment of each sentence), or aspect level (determining the sentiment of specific aspects or entities mentioned in the text)
  - challenges:
    - to represent the text data in a way that machine learning algorithms can understand
    - to deal with the nuances and complexities of human language, such as irony, and ambiguity, which can affect the accuracy of the predictions
- Machine translation: converting text from one language to another
- Named entity recognition: identifying and extracting named entities such as names, organizations, and locations from text
- Question answering: answering questions posed in natural language, such as the ones used by chatbots or virtual assistants
- Text summarization: summarizing long documents or articles to extract the most important information

continue ...


## Model

<img width="933" alt="截屏2023-04-13 下午12 10 40" src="https://user-images.githubusercontent.com/82795673/231834393-1c6dd5e2-ab7e-45ac-a591-94d34a32154c.png">
During pre-training, the model is trained on unlabeled data over different pre-training tasks. For finetuning, the BERT model is first initialized with the pre-trained parameters, and all of the parameters are fine-tuned using labeled data from the downstream tasks. Each downstream task has separate fine-tuned models, even though they are initialized with the same pre-trained parameters. A distinctive feature of BERT is its unified architecture across different tasks. There is minimal difference between the pre-trained architecture and the final downstream architecture.


<img width="733" alt="截屏2023-04-13 下午12 28 08" src="https://user-images.githubusercontent.com/82795673/231838072-e9cd96a0-280b-433b-87e1-97956d964a4f.png">

## Dataset & Method

1. emotional classification dataset (416809 rows and 2 columns)


|   Fig. 1  |   Fig. 2   |
| :-------: | :-------: |
| <img width="931" alt="截屏2023-04-12 下午8 49 21" src="https://user-images.githubusercontent.com/82795673/231626160-b49ce809-3482-487e-9839-68b819b9ecaa.png"> | <img width="165" alt="截屏2023-04-12 下午8 53 59" src="https://user-images.githubusercontent.com/82795673/231626453-00866996-59af-4329-839e-71fdc74a6720.png"> |

2. Yelp reviews
- training data (559999 rows × 2 columns)
- testing data (37999 rows × 2 columns)


### STEP I: Build an Emotion Classificatioin Model

1. First, we split the full data into training and validation sets. It then creates a dictionary that maps each emotion label to a unique integer index and a dictionary that maps each integer index back to its corresponding emotion label.

```python
tmp_path = "/content/drive/MyDrive/transformer_project/NLP-Transformer/NLP-BERT/"
with open(tmp_path + "merged_training.pkl", "rb") as tmp_file:
    full_data = pkl.load(tmp_file)
n_train = int(0.75*len(full_data))

train_data = full_data[:n_train]
valid_data = full_data[n_train:]

uniq_labels = list(
    sorted(list(pd.unique(train_data["emotions"]))))
labels_dict = dict([
    (uniq_labels[x], x) for x in range(len(uniq_labels))])
idx_2_label = dict([
    (x, uniq_labels[x]) for x in range(len(uniq_labels))])

```

2. we process the training corpus to form the vocabulary for the model. It tokenizes each comment in the corpus, updates a counter with the frequency of each token, and filters out tokens that occur less than ***min_count*** times. The resulting vocabulary includes additional tokens for padding, the start of the sentence, the end of the sentence, and unknown words.

```python
# Form the vocabulary. #
valid_corpus = list(valid_data["text"].values)
train_corpus = list(train_data["text"].values)

w_counter  = Counter()
max_length = 180
min_count  = 10
tmp_count  = 0
for tmp_text in train_corpus:
    tmp_text = tmp_text.replace("\n", " \n ").strip()
    
    tmp_tokens = [
        x for x in word_tokenizer(
            tmp_text.lower()) if x != ""]
    w_counter.update(tmp_tokens)
    
    tmp_count += 1
    if tmp_count % 25000 == 0:
        proportion = tmp_count / len(train_corpus)
        print(str(round(proportion*100, 2)) + "%", 
              "comments processed.")

vocab_list = sorted(
    [x for x, y in w_counter.items() if y >= min_count])
vocab_list = sorted(
    vocab_list + ["<cls>", "<eos>", "<pad>", "<unk>"])
vocab_size = len(vocab_list)
```

3. This code snippet sets up some parameters and initializes a BERT-based classification model and AdamW optimizer for training on an emotion classification task

```python
# Parameters. #
#the probability of applying dropout noise to the input data
p_noise = 0.10
#maximum gradient value allowed during training
grad_clip = 1.00
#maximum number of training steps
steps_max  = 10000
#the number of emotion classes
n_classes  = len(labels_dict)
batch_size = 256
sub_batch  = 64
batch_test = 128

hidden_size  = 256
warmup_steps = 5000
cooling_step = 500
display_step = 250
restore_flag = False  

# Define the classifier model. #
bert_model = bert.BERT_Classifier(
    n_classes, 3, 4, hidden_size, 
    4*hidden_size, word2idx, max_length)
bert_optim = tfa.optimizers.AdamW(
    beta_1=0.9, beta_2=0.98, 
    epsilon=1.0e-9, weight_decay=1.0e-4)

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    bert_model=bert_model, 
    bert_optim=bert_optim)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)
    
# Format the data before training
y_valid = [
    labels_dict[x] for x in valid_data["emotions"].values]
y_train = [
    labels_dict[x] for x in train_data["emotions"].values]

#the integer values for special tokens added to the vocabulary
cls_token = word2idx["<cls>"]
pad_token = word2idx["<pad>"]
unk_token = word2idx["<unk>"]
eos_token = word2idx["<eos>"]
```

4. we implement a training loop for a BERT-based model,

```python
# steps_max = 10000
while n_iter < steps_max:
    # Constant warmup rate
    step_val = float(max(n_iter+1, warmup_steps))**(-0.5)
    #a learning rate is calculated based on the step number and the warmup schedule
    learn_rate_val = float(hidden_size)**(-0.5) * step_val
    
    # select a random batch of training data
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    labels_array[:] = n_classes
    sentence_tok[:, :] = pad_token
    sentence_tok[:, 0] = cls_token
    for n_index in range(batch_size):
        ...
        # Generate noise. #
        n_input  = len(tmp_i_tok)
        n_decode = n_input + 1
        in_noisy = [unk_token] * n_input
        tmp_flag = np.random.binomial(1, p_noise, size=n_input)
        in_noise = [
            tmp_i_tok[x] if tmp_flag[x] == 1 else \
                in_noisy[x] for x in range(n_input)]
        ...
    
    # train on sub-batches of the data
    tmp_loss = sub_batch_train_step(
        bert_model, sub_batch, 
        sentence_tok, labels_array, bert_optim, 
        learning_rate=learn_rate_val, gradient_clip=grad_clip)
    
    # Increment the step. #
    n_iter += 1
    ckpt.step.assign_add(1)
    
    tot_loss += tmp_loss.numpy()
    if n_iter % display_step == 0:
        # Get the validation accuracy. #
        pred_labels = []
        for n_val_batch in range(n_val_batches):
            id_st = n_val_batch * batch_size
            if n_val_batch == (n_val_batches-1):
                id_en = num_valid
            else:
                id_en = (n_val_batch+1) * batch_size
            
            # Perform inference. #
            tmp_pred_labels = bert_model.infer(
                x_valid[id_st:id_en, :]).numpy()
            pred_labels.append(tmp_pred_labels)
        
        # Concatenate the predicted labels. #
        pred_labels = np.concatenate(
            tuple(pred_labels), axis=0)
        
        # Compute the accuracy. #
        accuracy = np.sum(np.where(
            pred_labels == y_valid, 1, 0)) / num_valid
        del pred_labels
        
        avg_loss = tot_loss / display_step
        tot_loss = 0.0
        elapsed_tm = (time.time() - start_tm) / 60
        ......
 ## It generates noise in the input data and trains the model to reconstruct the original input from the noisy version. 
 ## The training loop also includes a validation step to monitor the accuracy of the model
```

training and validation progress of the model:

<p align="center">
  <img src="https://user-images.githubusercontent.com/82795673/232249729-21c9fda5-3605-4c2a-aebf-85cf17bf0bbd.jpg" alt="training_loss" width="550"/>
</p>

after 2,000 iterations, the model has achieved a training loss of around 1.4, which indicates that the model is fitting the training data well

<p align="center">
  <img src="https://user-images.githubusercontent.com/82795673/232250364-c7a5a62b-2daf-4562-8d89-ef112c11577f.jpeg" alt="validation_accuracy" width="550"/>
</p>

validation accuracy is greater than 0.90, which suggests that the model is generalizing well to unseen data

### STEP 2: Train BERT Polarity Classifier

1. we first initialize a Counter() object to keep track of word frequencies, and then loop through each row of the data to clean and tokenize the text.

```python
# Define the function to clean the data
def clean_data(data_df, seq_len=9999):
    n_lines = len(data_df)
    
    # Process the data.
    
    #keep track of word frequencies
    w_counter = Counter()
    data_list = []
    for n_line in range(n_lines):
        data_line = data_df.iloc[n_line]
        data_text = data_line["text"]
        data_label = data_line["label"]
        
        # Clean the data a little. #
        data_text = data_text.replace("\n", " ")
        
        # Tokenize the words. #
        tmp_tokens = [
            x for x in wordpunct_tokenize(
                data_text.lower()) if x != ""]
                
        # update the word frequency with the tokens and appends a tuple of the label and tokens to the data_list variable
        w_counter.update(tmp_tokens)
        data_list.append((data_label, tmp_tokens))
        
        if (n_line+1) % 100000 == 0:
            percent_complete = round(n_line / n_lines * 100, 2)
            print(str(n_line), "rows", 
                  "(" + str(percent_complete) + "%) complete.")
    return data_list, w_counter
```

2. we clean the train_data and test_data from yelp_review_polarity dataset
```python
train_df = pd.read_csv(tmp_path + "train.csv")
train_df.columns = ["label", "text"]
# the input sequences will be truncated to a maximum of 100 tokens
maximum_seq_len  = 100

# Process the train dataset. #
print("Cleaning the train data.")
train_data, w_counter = clean_data(
    train_df, seq_len=maximum_seq_len)

print("Data formatted. Total of", 
      str(len(train_data)), "training samples.")

# Filter noise. #
min_count  = 10
word_vocab = list(sorted([
    word for word, count in \
    w_counter.items() if count >= min_count]))
word_vocab = \
    ["CLS", "UNK", "PAD", "EOS", "TRUNC"] + word_vocab
idx_2_word = dict([(
    x, word_vocab[x]) for x in range(len(word_vocab))])
word_2_idx = dict([(
    word_vocab[x], x) for x in range(len(word_vocab))])
```

3. After saving the processed training data, the word vocabulary, and the idx_2_word and word_2_idx dictionaries to a pickle file, we try to train BERT polarity classifier

```python
#part1:
bert_model = bert.BERT_Classifier(
    num_class, num_layers, num_heads, 
    hidden_size, ffwd_size, word_2_idx, 
    seq_length + 3, p_keep=prob_keep) #  ‘p_keep’ is the probability of keeping a neuron active during dropout
bert_optimizer = tfa.optimizers.AdamW(
    weight_decay=1.0e-4)
...

#part2: set up the parameters and settings for training the BERT model...
# Train the Transformer model
tmp_in_seq  = np.zeros(
    [batch_size, seq_length+3], dtype=np.int32)
tmp_out_lab = np.zeros([batch_size], dtype=np.int32)

# Warmup learning schedule. #
n_iter = ckpt.step.numpy().astype(np.int32)
if warmup_flag:
    step_min = float(max(n_iter, warmup_steps))**(-0.5)
    learning_rate = float(hidden_size)**(-0.5) * step_min
else:
    initial_lr = 0.001
    anneal_pow = int(n_iter / anneal_step)
    learning_rate = max(np.power(
        anneal_rate, anneal_pow)*initial_lr, 1.0e-5)
        
...
#part3: a training loop for a neural network and works by updating the network's weights based on the training data
# Update the neural network's weights
tot_loss = 0.0
start_tm = time.time()
while n_iter < maximum_iter:
    if warmup_flag:
        step_min = float(max(n_iter, warmup_steps))**(-0.5)
        learning_rate = float(hidden_size)**(-0.5) * step_min
    else:
        if n_iter % anneal_step == 0:
            anneal_pow = int(n_iter / anneal_step)
            learning_rate = max(np.power(
                anneal_rate, anneal_pow)*initial_lr, 1.0e-4)
    
    # Select a sample from the data. #
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    tmp_out_lab[:] = 0
    tmp_in_seq[:, :] = PAD_token
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_label = train_data[tmp_index][0]
        tmp_i_tok = train_data[tmp_index][1]
        
        ...
    
    # Set the training data. #
    tmp_input  = tmp_in_seq
    tmp_output = tmp_out_lab
    
    #The neural network is then trained on a sub-batch of the data, using the sub_batch_train_step function. 
    #The weights of the network are updated based on the loss calculated in the training step
    tmp_loss = sub_batch_train_step(
        bert_model, sub_batch, 
        tmp_input, tmp_output, bert_optimizer, 
        learning_rate=learning_rate, grad_clip=grad_clip)
    
    n_iter += 1
    ckpt.step.assign_add(1)
    
    tot_loss += tmp_loss.numpy()
    # For every display_step iteration, the accuracy of the network is calculated on the test data
    if n_iter % display_step == 0:
        # For simplicity, get the test accuracy #
        # instead of validation accuracy.       #
        pred_labels = []
        for n_val_batch in range(n_val_batches):
            id_st = n_val_batch * batch_test
            id_en = (n_val_batch+1) * batch_test
            
            if n_val_batch == (n_val_batches-1):
                curr_batch = num_test - id_st
            else:
                curr_batch = batch_test
            
            tmp_test_tokens = np.zeros(
                [curr_batch, seq_length+3], dtype=np.int32)
            
            tmp_test_tokens[:, :] = PAD_token
            tmp_test_tokens[:, 0] = CLS_token
            
            ...
                    
        # Compute the accuracy. #
        accuracy = np.sum(np.where(
            pred_labels == test_labels, 1, 0)) / num_test
        del pred_labels
        
        # keeps track of the total loss and the average loss per iteration
        end_tm = time.time()
        avg_loss = tot_loss / display_step
        tot_loss = 0.0
        elapsed_tm = (end_tm - start_tm) / 60
        
        ...   

```


4. Evaluation Metrics

we have got the evaluation report for the text classification model on emotional classification and Yelp reviews datasets

The final evaluation on the validation dataset of the emotion classification model is as follows:
```
              precision    recall  f1-score   support

           0       0.92      0.89      0.91     14320
           1       0.85      0.89      0.87     11949
           2       0.91      0.95      0.93     35285
           3       0.87      0.73      0.79      8724
           4       0.93      0.94      0.94     30161
           5       0.83      0.71      0.77      3764

    accuracy                           0.90    104203
   macro avg       0.89      0.85      0.87    104203
weighted avg       0.90      0.90      0.90    104203
```

- Explain: the emotion classification model achieves high precision, recall, and F1 scores for most of the classes, indicating that it performs well in identifying the different classes. However, the recall for classes 3 and 5 is relatively lower than the other classes, which means that the model may struggle to identify these classes. The accuracy score of 0.90 indicates that the model is accurate in its predictions, and the macro and weighted averages suggest that the model performs well overall.

The evaluation on the validation dataset of YELP polarity model is as follows:
```
              precision    recall  f1-score   support

           0       0.92      0.93      0.93     19000
           1       0.93      0.92      0.93     18999

    accuracy                           0.93     37999
   macro avg       0.93      0.93      0.93     37999
weighted avg       0.93      0.93      0.93     37999
```
- Explain: For YELP polarity model, the evaluation metrics are pretty high, and macro and weighted averages show that the model performs well in identifying both classes.


## Critical Analysis
1. advantage: emotion classification model and YELP polarity model performs well in identifying the different classes.
2. limitations and areas for improvement: While the BERT model has shown to be highly effective for a variety of natural language processing tasks, it still has some limitations and areas for improvement. One major limitation is the high computational cost required for training and fine-tuning the model, which can make it difficult to scale up to larger datasets or models. Additionally, while the model is highly effective at capturing contextual information, it may still struggle with certain types of linguistic phenomena such as negation or sarcasm.

## Links
datasets: https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/yelp_polarity.py

Paper: https://arxiv.org/abs/1810.04805v2

