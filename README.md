# Gaming-Text-Analysis
Predict gaming firms' financials using Youtube comments

Youtube Data API Documentation regarding comments: https://developers.google.com/youtube/v3/docs/comments

# "GetYoutubeComment.py"
Only extracting videos on the first page of search results.

  * Input:   keyword to search relevant videos.  Eg: 'Official Call of Duty: Infinite Warfare Reveal Trailer'  
             OR 
             a csv file with list of video game names
             
  * Output:  1) . Summary file 'overall_comments.csv' .  ['publish_time', 'video_ID', 'Title', 'viewCount',
                                      'likeCount', 'dislikeCount', 'favoriteCount', 'commentCount']
  2) . For each video: 'XGameName_comments.csv'    ['publish_time', 'Video ID', 'Title', 'Comment', 'updatedAt', 'likeCount']               
  
Before running "GetYoutubeComment.py":

## 0. Install prerequisite
('google-auth-oauthlib' requires sudo for reasons I don't know) 
```
pip install google-api-python-client google-auth google-auth-httplib2
sudo pip install google-auth-oauthlib
```

## 1. Project Setup for YouTube Data API
