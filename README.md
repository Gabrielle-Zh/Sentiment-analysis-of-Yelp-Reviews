# Sentiment-analysis-of-Yelp-Reviews

## Overivew

#### NLP application:

- Text classification and sentiment analysis: determining the topic and sentiment of a given text, for example, categorizing customer reviews as positive or negative
  - useful for analyzing customer feedback, social media posts, and reviews
  - performed at different levels, such as document level (determining the overall sentiment of a document), sentence level (determining the sentiment of each sentence), or aspect level (determining the sentiment of specific aspects or entities mentioned in the text)
  - challenges:
    - to represent the text data in a way that machine learning algorithms can understand
    - to deal with the nuances and complexities of human language, such as sarcasm, irony, and ambiguity, which can affect the accuracy of the predictions
- Machine translation: converting text from one language to another
- Named entity recognition: identifying and extracting named entities such as names, organizations, and locations from text
- Question answering: answering questions posed in natural language, such as the ones used by chatbots or virtual assistants
- Text summarization: summarizing long documents or articles to extract the most important information

continue ...


## Method

### Dataset

1. emotional classification dataset (416809 rows and 2 columns)


|   Fig. 1  |   Fig. 2   |
| :-------: | :-------: |
| <img width="931" alt="截屏2023-04-12 下午8 49 21" src="https://user-images.githubusercontent.com/82795673/231626160-b49ce809-3482-487e-9839-68b819b9ecaa.png"> | <img width="165" alt="截屏2023-04-12 下午8 53 59" src="https://user-images.githubusercontent.com/82795673/231626453-00866996-59af-4329-839e-71fdc74a6720.png"> |

2. Yelp reviews
- training data (559999 rows × 2 columns)
- testing data (37999 rows × 2 columns)


STEP 1: to build emotion classificatioin model
- initalize BERT and AdamW optimizer
- map the emotion labels to integer value (labels_dict dictionary) to train and validate the bert model
- perform inference
- xxxxxx



### Model

<img width="933" alt="截屏2023-04-13 下午12 10 40" src="https://user-images.githubusercontent.com/82795673/231834393-1c6dd5e2-ab7e-45ac-a591-94d34a32154c.png">

<img width="733" alt="截屏2023-04-13 下午12 28 08" src="https://user-images.githubusercontent.com/82795673/231838072-e9cd96a0-280b-433b-87e1-97956d964a4f.png">


## Code Demonstration

## Critical Analysis

## Links
datasets: https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/yelp_polarity.py

Paper: https://arxiv.org/abs/1810.04805v2

