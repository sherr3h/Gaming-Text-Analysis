# Gaming-Text-Analysis
Predict gaming firms' financials using Youtube comments

✴➡[Transmitter](https://drive.google.com/drive/folders/1SjHt-wRC7Cj-UbdV0WYDsXPpiHCuN3XS?usp=sharing) ⬅✴ to Group project Google drive

Youtube Data API Documentation regarding comments: https://developers.google.com/youtube/v3/docs/comments

# "GetYoutubeComment.py"
Due to Youtube API's daily data limit of 10000 units, only extract the first (few) videos on the first page of search results.

  * Input:    
     - keyword to search relevant videos.  Eg: 'Official Call of Duty: Infinite Warfare Reveal Trailer'  
     - OR a csv file with list of video game names
             
  * Output:  
     - Summary file 'overall_comments.csv' .  ['publish_time', 'video_ID', 'Title', 'viewCount',
                                      'likeCount', 'dislikeCount', 'favoriteCount', 'commentCount']
     - For each video: 'XGameName_comments.csv'    ['publish_time', 'Video ID', 'Title', 'Comment', 'updatedAt', 'likeCount']               
  
Before running "GetYoutubeComment.py":

### 0. Install prerequisite
(`google-auth-oauthlib` doesn't raise error when installing with sudo) 
```
pip install google-api-python-client google-auth google-auth-httplib2
sudo pip install google-auth-oauthlib
```

### 1. Setup YouTube Data API on Google developer

### 2. Run 


# "Textpreprocess.py"
Pre-processing and exploratory text analysis using `nltk` and `wordcloud` package.

# "MoodLearning.py"

# Todo
  * Complete corpus - Get all threads for trailers with an enormous amount of comments
  * More exploratory analysis
  * Find a suitable sentiment analysis instrument  - POMS?
  * Financial data. How to find a proper time frame for strategy?
