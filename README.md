# Gaming-Text-Analysis
Predict gaming firms' financials using Youtube comments

Youtube Data API Documentation regarding comments: https://developers.google.com/youtube/v3/docs/comments

# "GetYoutubeComment.py"
This code is so far very unorganised. 

Only extracting videos on the first page of search results (but that can easily be changed).

| Input  | keyword to search relevant videos | Eg: 'Official Call of Duty: Infinite Warfare Reveal Trailer'  |
| Output | comments.csv                      | ['Video ID', 'Title', 'Comment', 'publishedAt']               |


Before running "GetYoutubeComment.py":

## 0. Install dependencies 
('google-auth-oauthlib' requires sudo for reasons I don't know) 
```
pip install google-api-python-client google-auth google-auth-httplib2
sudo pip install google-auth-oauthlib
```

## 1. Project Setup for YouTube Data API
