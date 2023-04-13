import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# function to transform wordcloud
def transform_format(val):
    if val == 0:
        return 255
    else:
        return val

def main():
    # ignoring warnings
    warnings.filterwarnings("ignore")
    # creating pandas dataframe
    df = pd.read_csv('pamtag_robot_tags_v2.csv')

    # splitting on positive and negative reviews
    pos_df = (df[df['Sentiment'] == 'Positive'])
    neg_df = (df[df['Sentiment'] == 'Negative'])

    # positive sentiment wordcloud
    # putting all reviews into a single text blob
    pos_text = " ".join(review for review in pos_df['Text'])

    # create a stop word list
    stopwords = set(STOPWORDS)
    stopwords.update(["hotel", "room", "robot",
                      "stay", "one", "u"])

    happy_mask = np.array(Image.open("happy_face.png"))

    # transforming array into one that will work with the function
    transformed_happy_mask = np.ndarray((happy_mask.shape[0], happy_mask.shape[1]), np.int32)

    for i in range(len(happy_mask)):
        transformed_happy_mask[i] = list(map(transform_format, happy_mask[i]))

    # create and generate a word cloud image
    wordcloud = WordCloud(
        stopwords=stopwords,
        mask=transformed_happy_mask,
        colormap="summer",
        background_color="white",
        #max_words=200
    ).generate(pos_text)

    # Display word cloud
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    wordcloud.to_file("pos_wordcloud.png")

    # negative sentiment word cloud
    # putting all text into one blob
    neg_text = " ".join(review for review in neg_df['Text'])

    # create stopword list
    stopwords = set(STOPWORDS)
    stopwords.update(["hotel", "room", "robot",
                      "stay", "one", "u", "good"])

    sad_mask = np.array(Image.open("sad_face.png"))

    # transforming array into one that will work with the function
    transformed_sad_mask = np.ndarray((sad_mask.shape[0], sad_mask.shape[1]), np.int32)

    for i in range(len(sad_mask)):
        transformed_sad_mask[i] = list(map(transform_format, sad_mask[i]))

    # create and generate a wordcloud image
    wordcloud = WordCloud(
        stopwords=stopwords,
        mask=transformed_sad_mask,
        colormap="autumn",
        background_color="white",
        #max_words=300
    ).generate(neg_text)

    # display wordcloud
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    wordcloud.to_file("neg_wordlcoud.png")

if __name__ == '__main__':
    main()
