# Gaming-Text-Analysis
Predict entertainment firms' financials using Youtube comments

* ✴Download a subfolder (raw data) from Github:
```
svn checkout https://github.com/sherr3h/Gaming-Text-Analysis/trunk/working/Raw_Data_Studios/
svn checkout https://github.com/sherr3h/Gaming-Text-Analysis/trunk/working/Financial_Data/
```
* Youtube Data API Documentation about comments: https://developers.google.com/youtube/v3/docs/comments


# "GetYoutubeComment.py"
Due to Youtube API's daily data limit of 10000 units(1 request=1 unit), only extract the first (few) videos on the first page of search results.

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


