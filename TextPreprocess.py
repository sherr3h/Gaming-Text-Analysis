#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Pre-processing and Exploratory Analysis
@author: SH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import os
import io
from unidecode import unidecode
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

'''  **********  when first downloading nltk.corpus.stopwords  **********
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()
'''


def remove_stopw_review(review):
    '''
    :param
        review (string) :  a single piece of review
    :return:
        filtered_review (string) : a single piece of review, which has undergo:
            0) tokenize
            1) remove punctuation
            2) lemmatize
            3) remove stopwords
    '''

    word_tokens = word_tokenize(review.lower())
    res = " ".join(word_tokens)
    res = unidecode(res).replace('"', '').replace("'", '')
    no_digit_res = res.translate(str.maketrans('', '', string.digits))
    word_tokens_no_punc_list = no_digit_res.translate(str.maketrans('', '', string.punctuation)).split()
    lemmatized_sentence = [nltk.stem.WordNetLemmatizer().lemmatize(w, 'v') for w in word_tokens_no_punc_list]
    filtered_sentence = [w for w in lemmatized_sentence if not w in stop_words_add]
    filtered_review = " ".join(filtered_sentence)
    return filtered_review


def remove_stopw_df(input_df):
    new_text = input_df['Comment'].map(remove_stopw_review)
    output_col = [col for col in input_df.columns if col != 'Comment']
    #['publish_time', 'Video ID', 'Title', 'updatedAt', 'likeCount']
    output_df = input_df.loc[:, output_col]
    output_df.loc[:, 'Comment'] = new_text
    return output_df, new_text


def get_list_of_words(input_df):
    output_list = []
    for line in input_df.loc[:,'Comment']:
        try:
            output_list.extend([str(x) for x in line.split()])
        except:
            pass
    return output_list


def remove_rare_word(review):
    word_tokens = word_tokenize(review)
    filtered_sentence = [w for w in word_tokens if w in common_wordset]
    filtered_review = " ".join(filtered_sentence)
    return filtered_review


def remove_rarew_df(input_df):
    new_text = input_df['Comment'].map(remove_rare_word)
    output_df = input_df.loc[:, ['Video ID', 'Title', 'updatedAt']]
    output_df.loc[:, 'Comment'] = new_text
    return output_df


def pre_process_1(raw_comment_df):
    # remove stopwords
    process_comment_df_1, new_text = remove_stopw_df(raw_comment_df)
    print('Stopwords removed.')

    # get a list of word
    comment_word_list = get_list_of_words(process_comment_df_1)  # Battle field V no. of words 43820
    print('No. of total words in corpus:', len(comment_word_list))

    #  Get bag of words
    comment_wordset = set(comment_word_list)  # 5957 #pikachu: 1748
    print('No. of distinct words:', len(comment_wordset))

    # Get word distribution as NLTK FreqDist object
    freq_dist = nltk.FreqDist(comment_word_list)

    # Get bag of words by removing rare strings (occurance <= 1)
    common_words = list(filter(lambda x: x[1] > 1, freq_dist.items()))
    common_wordset = set(dict(common_words).keys())
    print('Size of bag of words:', len(common_wordset))

    return process_comment_df_1, freq_dist, common_wordset, comment_word_list


def generate_word_cloud(comment_word_list, mask=None,
                            output_file="try_word_cloud.png",
                            maxword=200,
                            stopwords_set=set(['like']),
                            bg_color="white",
                            contour_width=5,
                            contour_color='gold', **kwargs):
    from os import path
    from PIL import Image
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

    comment_word_str = ' '.join(comment_word_list)  # 34057
    if mask is not None:
        pikachu_mask = np.array(Image.open(mask))
        wc = WordCloud(background_color=bg_color,
                           max_words=maxword, mask=pikachu_mask,
                           stopwords=stopwords_set,  # set(['Pokemon','game'])
                           contour_width=contour_width,
                           contour_color=contour_color, **kwargs)  # pikachu: 'gold'
    else:
        wc = WordCloud(background_color=bg_color,
                           max_words=maxword,
                           stopwords=stopwords_set,  # set(['Pokemon','game'])
                           contour_width=contour_width,
                           contour_color=contour_color, **kwargs)

    wc.generate(comment_word_str)
    plt.figure(figsize=[20, 10])
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    wc.to_file(output_file)
    return

if __name__ == '__main__':

    '''************ Input ************'''

    # region input
    Combine_all_file = False
    file_name = 'combined_raw.csv' #'Battlefield V_comments.csv'
    os.getcwd()
    folder_path = "FINA4350/All-Raw-Data"
    if folder_path is not None:
        os.chdir(folder_path)


    # endregion input
    '''************ End of Input ************'''

    if not Combine_all_file:
        raw_comment_df = pd.read_csv(file_name, lineterminator='\n', error_bad_lines=False) #, encoding='utf-8'
        raw_comment_df.Comment = raw_comment_df.Comment.astype(str)
    else:
        #### Section 0. Create corpus, combine all files in the folder
        import glob
        extension = 'csv'
        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

        ''''#    for debugging
        import csv
        with open(r'combined_raw.csv', 'r') as f:
            reader = csv.reader(f)
            linenumber = 1
            try:
                for row in reader:
                    linenumber += 1
            except Exception as e:
                print(("Error line %d: %s " % (linenumber, str(type(e)))))
        '''
        combined_csv = pd.concat([pd.read_csv(f, lineterminator='\n') for f in all_filenames])
        print(combined_csv.shape)  # (96164, 7)

        raw_comment_df = combined_csv
        raw_comment_df.Comment = raw_comment_df.Comment.astype(str)

    raw_comment_df['OriTextLen'] = raw_comment_df['Comment'].str.len()

    ########## ########## ########## ########## ##########
    # Section 1.  Remove stop words and rare words:
    #    Stop words are a pre-defined set
    #    Rare words are defined as frequency <2
    ########## ########## ########## ########## ##########

    stop_words = set(stopwords.words('english'))
    stop_words_deutsch = set(stopwords.words('german'))
    '''
    stop_words
    {‘ourselves’, ‘hers’, ‘between’, ‘yourself’, ‘but’, ‘again’, ‘there’, ‘about’, ‘once’, 
    ‘during’, ‘out’, ‘very’, ‘having’, ‘with’, ‘they’, ‘own’, ‘an’, ‘be’, ‘some’, ‘for’, ‘do’, 
    ‘its’, ‘yours’, ‘such’, ‘into’, ‘of’, ‘most’, ‘itself’, ‘other’, ‘off’, ‘is’, ‘s’, ‘am’, 
    ‘or’, ‘who’, ‘as’, ‘from’, ‘him’, ‘each’, ‘the’, ‘themselves’, ‘until’, ‘below’, ‘are’, 
    ‘we’, ‘these’, ‘your’, ‘his’, ‘through’, ‘don’, ‘nor’, ‘me’, ‘were’, ‘her’, ‘more’, ‘himself’, 
    ‘this’, ‘down’, ‘should’, ‘our’, ‘their’, ‘while’, ‘above’, ‘both’, ‘up’, ‘to’, ‘ours’, ‘had’, 
    ‘she’, ‘all’, ‘no’, ‘when’, ‘at’, ‘any’, ‘before’, ‘them’, ‘same’, ‘and’, ‘been’, ‘have’, ‘in’, 
    ‘will’, ‘on’, ‘does’, ‘yourselves’, ‘then’, ‘that’, ‘because’, ‘what’, ‘over’, ‘why’, ‘so’, 
    ‘can’, ‘did’, ‘not’, ‘now’, ‘under’, ‘he’, ‘you’, ‘herself’, ‘has’, ‘just’, ‘where’, ‘too’, 
    ‘only’, ‘myself’, ‘which’, ‘those’, ‘i’, ‘after’, ‘few’, ‘whom’, ‘t’, ‘being’, ‘if’, ‘theirs’, 
    ‘my’, ‘against’, ‘a’, ‘by’, ‘doing’, ‘it’, ‘how’, ‘further’, ‘was’, ‘here’, ‘than’}
    '''

    stop_words_add =  stop_words.union(stop_words_deutsch)
    stop_words_add.update(["'",'let','da','der','du','dem','es','ds','oh','na',
                           'look','still','say','want','think','know','need','see',
                           'u','m','d','n','re','nt','go','come','would','ve','get', 'give',
                           'us','also','really','one','could','even','much','always','take','make'])

    process_comment_df_1, freq_dist, common_wordset, comment_word_list = pre_process_1(raw_comment_df)
    # remove rare strings (occurance <= 1)
    process_comment_df_2 = remove_rarew_df(process_comment_df_1)

    output_name = file_name.replace(".csv", "_processed.csv")

    process_comment_df_2.to_csv(output_name, sep=',', index=False, encoding='utf-8')

    ########## ########## ########## ##########
    #### Section 2. Visualization

    # region 2.0) Plot the number of appearance of top words

    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(8, 6))
    freq_dist.plot(40, cumulative=False)
    word_dist_name = file_name.replace(".csv", "_word_dist.jpg")
    plt.savefig(word_dist_name)
    # endregion 2.0) Plot the number of appearance of top words

    # region 2.1) Customize shape word cloud

    mask_img = "FINA4350/cod_logo.jpg"

    generate_word_cloud(comment_word_list, mask=mask_img,
                        output_file="try_word_cloud.png",
                        maxword=200,
                        #stopwords_set=set(['like']),
                        bg_color="white",
                        contour_width=5,
                        contour_color='gold'
                        )

    # endregion 2.1) Customize shape word cloud
    ########## ########## ########## ##########


###### one-time code to generate corpus-summary graphs ######
    # as of 27/10/19
    # No. of total words in corpus: 594699
    # No. of distinct words: 43287
    # Size of bag of words: 16742
    
    if Combine_all_file:
        combined_csv_new = process_comment_df_2

        combined_csv_new['NewTextLen'] = combined_csv_new['Comment'].str.len()
        mean_comment_len = sum(combined_csv_new.NewTextLen)/len(combined_csv_new.NewTextLen)
        max(combined_csv_new.NewTextLen) #7679
        combined_text_q = [combined_csv_new.NewTextLen.quantile(0.25),# 12.0
                           combined_csv_new.NewTextLen.quantile(0.5),  # 23.0
                           combined_csv_new.NewTextLen.quantile(0.75),  # 42
                           combined_csv_new.NewTextLen.quantile(0.9)]  # 74

        combined_csv_textlen_plot = combined_csv_new.NewTextLen[combined_csv_new.NewTextLen<100]

        ###### Plot corpus summary statistics ######
        plt.rcParams.update({'font.size': 15})

        fig, ax = plt.subplots()
        n, bins, patches = plt.hist(combined_csv_textlen_plot, 25, color='#6495ED', alpha=0.55, zorder=2, rwidth=1,
                                    histtype='stepfilled')
        ax.axvline(combined_text_q[0], dashes=[5, 2], color='#4169E1', linewidth=2)
        ax.axvline(combined_text_q[1], dashes=[5, 2], color='#4169E1', linewidth=2)
        ax.axvline(combined_text_q[2], dashes=[5, 2], color='#4169E1', linewidth=2)
        ax.axvline(combined_text_q[3], dashes=[5, 2], color='#4169E1', linewidth=2)
        ax.text(0.7, 0.4, 'average word count = ' + str(round(mean_comment_len,
                                                                    2)) + '\n max len of comment = 7679\n\n Vertical lines represent\n 10%,50%,75%,90% quantiles.',
                verticalalignment='bottom', horizontalalignment='center',
                transform=ax.transAxes,
                color='#0000CD')
        plt.title("Histogram of Youtube comments length (processed) ")
        plt.xlabel('Wordcount of a comment')
        plt.ylabel('No of comment')
        plt.savefig("comment_wordcount_hist.png")
        plt.show()

