#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Pre-processing and Exploratory Analysis
@author: Sherry He
This file performs several tasks:
    1. Text pre-processing using nltk
    2. Deadling with misspellings, unrecognisable strings and names
    2. LDA, Bag-of-word and tf-idf model using gensim
    3. Visualisation, including: word cloud, word distribution,
                      polarity and subjectivity,  movie timeline,
                      latent factor analysis
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
import pickle
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from textblob import TextBlob
import pkg_resources
import symspellpy
from symspellpy import SymSpell, Verbosity
from names_dataset import NameDataset

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


def pre_process(raw_comment_df):
    # remove stopwords
    process_comment_df_1, new_text = remove_stopw_df(raw_comment_df)
    print('Stopwords removed.')

    # get a list of word
    comment_word_list = get_list_of_words(process_comment_df_1)
    print('No. of total words in corpus:', len(comment_word_list))

    #  Get bag of words
    comment_wordset = set(comment_word_list)
    print('No. of distinct words:', len(comment_wordset))

    # Get word distribution as NLTK FreqDist object
    freq_dist = nltk.FreqDist(comment_word_list)

    # Get bag of words by removing rare strings (occurrence <= 4)
    common_words = list(filter(lambda x: x[1] > 4, freq_dist.items()))
    common_wordset = set(dict(common_words).keys())
    print('Size of bag of words:', len(common_wordset))

    return process_comment_df_1, freq_dist, common_wordset, comment_word_list


def is_name(word):
    m = NameDataset()
    return m.search_first_name(word) & m.search_last_name(word)


def spellcheck_keep_name(input_term):
    if len(input_term) == 0:
        return ' '
    # max edit distance per lookup (per single word, not per whole input string)
    suggestions = sym_spell.lookup_compound(input_term, max_edit_distance=2)
    '''
    # display suggestion term, edit distance, and term frequency
    for suggestion in suggestions:
        print(suggestion)'''
    suggested = suggestions[0].term.split()
    inputed = input_term.split()
    min_len = min(len(suggested), len(inputed))
    ret = []
    for i in range(min_len):
        if is_name(inputed[i]):
            ret.append(inputed[i])
        elif inputed[i] in english_words:
            ret.append(suggested[i])
    #ret = [inputed[i] if is_name(inputed[i]) else suggested[i] for i in range(min_len)]
    return ' '.join(ret)


def load_words():
    #Source: https://github.com/dwyl/english-words
    with open('words_alpha.txt') as word_file:
        valid_words = set(word_file.read().split())
    return valid_words


def generate_word_cloud(comment_word_list, mask=None,
                            output_file="try_word_cloud.png",
                            maxword=200,
                            stopwords_set=set(['']),
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
    plt.figure(figsize=(20, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    wc.to_file(output_file)
    return

if __name__ == '__main__':

    '''************ Input ************'''
    Combine_all_file = True
    file_name = 'combined_raw.csv' #'Battlefield V_comments.csv'
    current_path = os.getcwd()
    print(current_path)
    folder_path = "FINA4350/Raw_Data_Studios" #FINA4350/

    if folder_path is not None and current_path !='/Users/sh/PycharmProjects/myProject/FINA4350/Raw_Data_Studios':
        os.chdir(folder_path)

    Tickers = ['CMCSA', 'DIS', 'DIS_Marvel', 'LGF', 'SNE', 'TFCF', 'TWX', 'VIA']
    '''************ End of Input ************'''

    stop_words = set(stopwords.words('english'))
    stop_words_deutsch = set(stopwords.words('german'))
    stop_words_add = stop_words.union(stop_words_deutsch)
    stop_words_add.update(["'", 'da', 'der', 'du', 'dem', 'es', 'ds', 'oh', 'na', 'ca',
                           'look', 'still', 'say', 'want', 'think', 'mean', 'know', 'need', 'see', 'let',
                           'u', 'm', 'd', 'n', 're', 'nt', 'go', 'come', 'would', 've', 'get', 'give',
                           'us', 'also', 'really', 'one', 'could', 'even', 'much', 'always', 'take',
                           'make', 'ever', 'fuck'])
    corpus_dict = {}
    raw_corpus_dict = {}
    corpus_word_list = []

    # this for loop perform text pre-processing for all movies under each ticker
    # the results for each ticker are saved as a dictionary {ticker: data frame}
    # this dictionary is later used for bag-of-words and td-idf model

    for tk in Tickers:
        print(tk)
        if tk == "CMCSA":
            os.chdir(tk)
        else:
            os.chdir("../" + tk)

        if not Combine_all_file:
            raw_comment_df = pd.read_csv(file_name, lineterminator='\n', error_bad_lines=False) #, encoding='utf-8'
            raw_comment_df.Comment = raw_comment_df.Comment.astype(str)
        else:
            #### Section 0. Create raw corpus, combine all files in the folder
            import glob
            extension = 'csv'
            all_filenames = [i for i in glob.glob("*.{}".format(extension))]
            print("No of movie:", len(all_filenames))

            '''
            # For Debugging
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

            combined_csv = pd.concat([pd.read_csv(f, lineterminator='\n') for f in all_filenames], sort=True)
            print(combined_csv.shape)

            raw_comment_df = combined_csv
            raw_comment_df.Comment = raw_comment_df.Comment.astype(str)

        raw_comment_df['OriTextLen'] = raw_comment_df['Comment'].str.len()

        ########## ########## ########## ########## ##########
        # Section 1.  Remove stop words and rare words:
        #    Stop words are from a pre-defined set
        #    Rare words are defined as frequency <2
        ########## ########## ########## ########## ##########

        # remove stopwords
        process_comment_df_1, new_text = remove_stopw_df(raw_comment_df)
        print('Stopwords removed.')

        # get a list of word
        comment_word_list = get_list_of_words(process_comment_df_1)
        print('No. of total words in comments of', tk, len(comment_word_list))

        raw_corpus_dict[tk] = process_comment_df_1
        corpus_word_list += comment_word_list
    # end of for loop

    #region construct overall corpus
    avrg_len = 0
    for v in raw_corpus_dict.values():
        avrg_len += v.OriTextLen.mean()
    avrg_len / len(raw_corpus_dict.keys()) # 77.1031

    file = open('raw_corpus_dict', 'wb')
    pickle.dump(raw_corpus_dict, file)
    file.close()

    print('No. of total words in whole corpus:', len(corpus_word_list)) #16887477
    #  Get bag of words
    comment_wordset = set(corpus_word_list)
    print('No. of distinct words in whole corpus:', len(comment_wordset)) #442525

    # Get word distribution as NLTK FreqDist object
    corpus_freq_dist = nltk.FreqDist(corpus_word_list)

    # Get bag of words by removing rare strings (occurrence <= 9)
    common_words = list(filter(lambda x: x[1] > 9, corpus_freq_dist.items()))
    common_wordset = set(dict(common_words).keys())
    print('Size of corpus bag of words:', len(common_wordset)) #37815
    # endregion construct overall corpus

    # this for loop construct bag-of-word and wordcloud for all movies under each ticker
    # the results for each ticker are saved as a dictionary {ticker: data frame}
    # this dictionary is later used for bag-of-words and td-idf model
    english_words = load_words()

    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
    # term_index is the column of the term and count_index is the
    # column of the term frequency
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    os.chdir("../")
    spellcheck_corpus_dict={}
    for tk in Tickers:
        # remove rare strings (occurrence <= 9)
        process_comment_df_2 = remove_rarew_df(raw_corpus_dict[tk])
        process_comment_df_2 = process_comment_df_2.reset_index(drop=True)

        # spellcheck
        process_comment_df_3 = process_comment_df_2.copy(deep=True)
        spellcheck_comment = process_comment_df_2.Comment.map(spellcheck_keep_name)
        process_comment_df_3.Comment = spellcheck_comment

        # textblob sentiment
        polarity = pd.Series([TextBlob(c).sentiment.polarity for c in spellcheck_comment], index=spellcheck_comment.index)
        subjectivity = pd.Series([TextBlob(c).sentiment.subjectivity for c in spellcheck_comment])
        process_comment_df_3['polarity'] = polarity
        process_comment_df_3['subjectivity'] = subjectivity

        if tk == "DIS_Marvel":
            spellcheck_corpus_dict[tk].loc[:, "Ticker"] = "DIS"
        else:
            spellcheck_corpus_dict[tk].loc[:,"Ticker"] = tk

        print('Finished spell-checking and sentiment analysis', tk)

        output_name = file_name.replace(".csv", "_" + tk + "_processed.csv")

        process_comment_df_3.to_csv(output_name, sep=',', index=False, encoding='utf-8')

        spellcheck_corpus_dict[tk] = process_comment_df_3

    file = open('spellcheck_corpus_dict', 'wb')
    pickle.dump(spellcheck_corpus_dict, file)
    file.close()

    with open(r"corpus_dict", "rb") as input_file:  # no spellcheck, sentiment
        corpus_dict = pickle.load(input_file)

    # aggregate comments of each movie to overall file
    output_df = pd.read_csv("overall_comments_24_Nov_clean.csv", lineterminator='\n',
                            error_bad_lines=False)  # , encoding='utf-8'
    del output_df['Ticker\r']

    all_df = pd.concat([v for k, v in spellcheck_corpus_dict.items()]) #(2525916, 7)
    all_df_SAmean = all_df.groupby(['Video ID','Ticker']).mean().reset_index()
    all_df_SAstd = all_df.groupby(['Video ID']).std().rename(columns={"polarity": "polarity_std", "subjectivity": "subjectivity_std"}).reset_index()
    output_df1 = output_df.rename(columns={"video_ID": "Video ID"})
    output_df2 = pd.merge(output_df1, all_df_SAmean, on='Video ID', how='inner')
    output_df2 = pd.merge(output_df2, all_df_SAstd, on='Video ID', how='inner') # (384, 15)
    output_df2.to_csv("overall_SA.csv", sep=',', index=False, encoding='utf-8')

    #output_df2 = pd.read_csv("overall_SA.csv", lineterminator='\n',
    #                        error_bad_lines=False)

    
    ########## ########## ########## ##########
    #### Section 2. Exploratory Analysis and Visualization

    # 2.0 Principal Component Analysis

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    features = ['viewCount', 'likeCount',
       'dislikeCount', 'commentCount','polarity', 'subjectivity',]
    # Separating out the features
    x = output_df2.loc[:, features].values
    # Separating out the target
    y = output_df2.loc[:, ['Ticker']].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, output_df2[['Ticker']]], axis=1)
    print(pca.explained_variance_ratio_)

    targets = output_df2.Ticker.unique()
    cdict = {'CMCSA': 'blue', 'SNE': 'red', 'TWX': 'purple', 'LGF': 'brown', 'DIS': 'pink', 'TFCF': 'olive',
             'VIA': 'cyan'}

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    for tk in output_df2.Ticker.unique():
        indicesToKeep = finalDf['Ticker'] == tk
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=cdict[tk]
                   , s=50)
    for i, txt in enumerate(output_df2.Movie):
        ax.annotate(txt, (finalDf.loc[i, 'principal component 1'], finalDf.loc[i, 'principal component 2']))

    ax.legend(targets)
    ax.grid()
    plt.show()
    plt.savefig("PCA2.jpg")

    # region 2.1) Plot the number of appearance of top words
    scatter_x = output_df2.polarity
    scatter_y = output_df2.subjectivity
    group = output_df2.Ticker

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for g in np.unique(group):
        ix = group == g
        ax.scatter(scatter_x[ix], scatter_y[ix], alpha=0.8,  c=cdict[g], edgecolors='none',label=g, s=30)
    #plt.title('Matplot scatter plot')
    plt.xlabel('polarity')
    plt.ylabel('subjectivity')
    plt.legend(loc=2)
    plt.savefig("pol_vs_sub.jpg")
    plt.show()
    # endregion 2.1) Plot the number of appearance of top words

    
    for tk in Tickers:
        # region 2.2) Plot the number of appearance of top words
        freq_dist = nltk.FreqDist(corpus_dict[tk])

        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize=(16, 12))
        freq_dist.plot(40, cumulative=False)
        word_dist_name = file_name.replace(".csv", "_"+tk+"_word_dist.jpg")
        plt.savefig(word_dist_name)
        # endregion 2.2) Plot the number of appearance of top words

        # region 2.3) Customize shape word cloud
        comment_word_list = get_list_of_words(corpus_dict[tk])

        mask_img = None

        generate_word_cloud(comment_word_list, mask=mask_img,
                            output_file=tk+"_word_cloud.png",
                            maxword=200,
                            stopwords_set=set(['like']),
                            bg_color="white",
                            contour_width=5,
                            contour_color='gold'
                            )

        # endregion 2.3) Customize shape word cloud
        ########## ########## ########## ##########

    #region 2.4) LDA from Gensim
    import gensim
    import gensim.corpora as corpora
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel

    # Plotting tools
    import pyLDAvis
    import pyLDAvis.gensim  # don't skip this
    import matplotlib.pyplot as plt

    #create gensim dictionary

    if Combine_all_file:
        os.chdir("../")
        #corpus_dict = {}
        all_studio_comment_list = []
        corpus_dict = {}
        for tk in Tickers:
            print(tk)
            '''
            if tk == "CMCSA":
                os.chdir(tk)
            else:
                os.chdir("../" + tk)
            #print =(output_name)
            '''
            output_name = "combined_raw_" + tk + "_processed.csv"
            corpus_dict[tk] = pd.read_csv(output_name, lineterminator='\n', error_bad_lines=False)
            corpus_dict[tk].Comment = corpus_dict[tk].Comment.astype(str)

            comment_list = []
            for text in corpus_dict[tk].Comment:
                comment_list += text.split(' ')
            all_studio_comment_list.append(comment_list)

        # Create Dictionary
        id2word = Dictionary(all_studio_comment_list)

        # Convert to vector corpus
        comment_vectors = [id2word.doc2bow(text) for text in all_studio_comment_list]

        # View
        print(comment_vectors[:1])

        # Build LDA model
        lda_model = gensim.models.ldamodel.LdaModel(corpus=comment_vectors,
                                                    id2word=id2word,
                                                    num_topics=15,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=30,
                                                    alpha='auto',
                                                    per_word_topics=True)
        from pprint import pprint
        # Print the Keyword in the 10 topics
        pprint(lda_model.print_topics(num_topics=15, num_words=40))
        doc_lda = lda_model[comment_vectors]

        # Compute Perplexity
        print('\nPerplexity: ',
              lda_model.log_perplexity(comment_vectors))  # a measure of how good the model is. lower the better.
        #Perplexity:  -7.675805717668594

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=all_studio_comment_list, dictionary=id2word,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)

        #Coherence Score:  0.28624721848288204

        # Visualize the topics
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim.prepare(lda_model, comment_vectors, id2word, mds='mmds')
        #pyLDAvis.show(vis)
        pyLDAvis.save_html(vis, 'lda_t15_w40.html')

        mallet_path = 'mallet-2.0.8/bin/mallet'  # update this path
        ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=comment_vectors, num_topics=20, id2word=id2word)

        # Show Topics
        pprint(ldamallet.show_topics(formatted=False))

        #endregion 2.4) LDA from Gensim

        #########
        # region 2.5) Build tfidf model
        comment_tfidf = TfidfModel(comment_vectors)

        top_words = np.sort(np.array(comment_tfidf[comment_vectors[0]], dtype=[('word', int), ('score', float)]),
                            order='score')[::-1] # reverse sort..
        #[(comment_dict[word], score) for word, score in top_words]

        # keeps only elements that return True
        new_top_words = np.array([(word,score) for word, score in top_words if score < 0.0001] , dtype=[('word', int), ('score', float)])

        from os import path
        from PIL import Image
        from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

        wc = WordCloud(background_color="white", mask=None, random_state=5, max_words=2000)
        wc.fit_words(dict([(comment_dict[word], score) for word, score in new_top_words]))

        plt.figure(figsize=[20, 10])
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        wc.to_file(Tickers[0]+"tfidf_word_cloud.png")
        
         # endregion 2.5) Build tfidf model

    # region 2.6) Word Distribution
    ###### summarize corpus for each studio ######
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
                                                                    2)) + '\n max len of comment = 7679\n\n Vertical lines represent\n 25%,50%,75%,90% quantiles.',
                verticalalignment='bottom', horizontalalignment='center',
                transform=ax.transAxes,
                color='#0000CD')
        plt.title("Histogram of Youtube comments length (processed) ")
        plt.xlabel('Wordcount of a comment')
        plt.ylabel('No of comment')
        plt.savefig("comment_wordcount_hist.png")
        plt.show()

    # endregion 2.6) Word Distribution
    
    # region 2.7) Movie release timeline
    #### draw a timeline for each movie production company ###

    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import matplotlib.dates as mdates
    import pytz
    from datetime import datetime

    file_name = 'overall_comments_24_Nov_clean.csv' #'Battlefield V_comments.csv'
    os.getcwd()
    folder_path = None #
    if folder_path is not None:
        os.chdir(folder_path)

    overall_comment_df = pd.read_csv(file_name, lineterminator='\n', error_bad_lines=False)  # , encoding='utf-8'

    overall_comment_df.ProductionFirm.unique() #'20th Century FOX', 'Disney', 'Lionsgate', 'Marvel', 'Paramount',
                                                #'Sony', 'Universal', 'Warner Bros'

    studio_dict = {}

    for studio in overall_comment_df.ProductionFirm.unique():
        studio_dict[studio] = overall_comment_df.loc[overall_comment_df.ProductionFirm == studio]

    plt.rcParams.update({'font.size': 9})
    before_date = pd.to_datetime(' 2018-05-31')
    before_date = pytz.utc.localize(before_date)

    for studio, studio_df in studio_dict.items():
        print(studio)

        dates = pd.to_datetime(studio_df.publish_time)
        dates_filtered = dates.loc[dates<before_date]
        names = studio_df.Movie.loc[dates_filtered.index]
        print("Number of movies of", studio, names.shape[0])

        levels = np.tile([-5, 5, -3, 3, -1, 1],
                     int(np.ceil(len(dates_filtered) / 6)))[:len(dates_filtered)]

        # Create figure and plot a stem plot with the date
        fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
        ax.set(title= studio + " release dates\nNumber of movies: " + str(names.shape[0]))

        markerline, stemline, baseline = ax.stem(dates_filtered, levels,
                                                 linefmt="C3-", basefmt="k-",
                                                 use_line_collection=True)

        plt.setp(markerline, mec="k", mfc="w", zorder=3)

        # Shift the markers to the baseline by replacing the y-data by zeros.
        markerline.set_ydata(np.zeros(len(dates_filtered)))

        # annotate lines
        vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
        for d, l, r, va in zip(dates_filtered, levels, names, vert):
            ax.annotate(r, xy=(d, l), xytext=(0, - l * 40), rotation=90, # - np.sign(l) * 35
                        textcoords="offset points", va=va, ha="right")

        # format xaxis with 4 month intervals
        ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=4))
        ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        # remove y axis and spines
        ax.get_yaxis().set_visible(False)
        for spine in ["left", "top", "right"]:
            ax.spines[spine].set_visible(False)

        ax.margins(y=0.1)

        plt.savefig(studio+"_Timeline.png")
        plt.show()
    # endregion 2.7) Movie release timeline
    
    ####### End of code ######
    
    '''
    CMCSA
    No of movie: 35
    (227377, 6)
    No. of total words in comments of CMCSA 1567061
    DIS
    No of movie: 39
    (328815, 6)
    No. of total words in comments of DIS 1994554
    DIS_Marvel
    No of movie: 17
    (243656, 6)
    No. of total words in comments of DIS_Marvel 1494941
    LGF
    No of movie: 51
    (137476, 6)
    No. of total words in comments of LGF 925818
    SNE
    No of movie: 34
    (220672, 6)
    No. of total words in comments of SNE 1477550
    TFCF
    No of movie: 53
    (276790, 6)
    No. of total words in comments of TFCF 1724550
    TWX
    No of movie: 73
    (825751, 6)
    No. of total words in comments of TWX 5815921
    VIA
    No of movie: 95
    (265379, 6)
    No. of total words in comments of VIA 1887082
    '''
