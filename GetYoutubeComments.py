import csv
import os
import pickle
import pandas as pd
import string

import google.oauth2.credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# The CLIENT_SECRETS_FILE variable specifies the name of a file that contains
# the OAuth 2.0 information for this application, including its client_id and
# client_secret.
CLIENT_SECRETS_FILE = "client_secret.json"

# This OAuth 2.0 access scope allows for full read/write access to the
# authenticated user's account and requires requests to use an SSL connection.
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'

#*** do not change this authentication function ***
def get_authenticated_service():
    credentials = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            credentials = pickle.load(token)
    #  Check if the credentials are invalid or do not exist
    if not credentials or not credentials.valid:
        # Check if the credentials have expired
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            credentials = flow.run_console()

        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(credentials, token)

    return build(API_SERVICE_NAME, API_VERSION, credentials=credentials)


def get_video_comments(service, **kwargs):
    comments = []
    results = service.commentThreads().list(**kwargs).execute()
    i = 0
    print("\nTotal comments: {0} \nResults per page: {1}".format(results['pageInfo']['totalResults'],
                                                              results['pageInfo']['resultsPerPage']))
    print("Example output per comment item, snippet")
    print(results['items'][0]['snippet'].keys())
    '''
    if second_time_download:
        k = 0
        while results and k < 99: #battlefield5: 589 
            try:
                if 'nextPageToken' in results:
                    #print('nextPageToken (already extracted):', results['nextPageToken'])
                    kwargs['pageToken'] = results['nextPageToken']
                    results = service.commentThreads().list(**kwargs).execute()
                    k += 1
                else:
                    break
            except:
                print("**Error on the {0} th page, nextPageToken: {1}".format(k, results['nextPageToken']))
                pass
    
        print("Starting extracting from page", k)
    '''
    while results and i < max_comment_per_video:  # commentThreads() maxResults = 100
        for item in results['items']:
            comment = [item['snippet']['topLevelComment']['snippet']['textDisplay'],
                       item['snippet']['topLevelComment']['snippet']['updatedAt'], #publishedAt
                       item['snippet']['topLevelComment']['snippet']['likeCount']]
            comments.append(comment)
            i += 1

        # Check if another page exists
        try:
            if 'nextPageToken' in results:
                #print('nextPageToken', results['nextPageToken'])
                kwargs['pageToken'] = results['nextPageToken']
                results = service.commentThreads().list(**kwargs).execute()
            else:
                break
        except:
            print("**Error on the {0} th page, nextPageToken: {1}".format(i, results['nextPageToken']))
            pass

    return comments


def get_video_stat(service, **kwargs):
    stats = []
    results = service.videos().list(**kwargs).execute()

    for item in results['items']:
        stat = [item['statistics']['viewCount'],
                    item['statistics']['likeCount'],
                    item['statistics']['dislikeCount'],
                    item['statistics']['favoriteCount'],
                    item['statistics']['commentCount']]
        stats.append(stat)

    return stats


def write_to_csv(Youtube_res_list, fname, comment_file=True, append_write='w'):
    '''
    :param append_write:  string, {'a','w'}, choose append option 'a' only if file already exists
    '''
    if not os.path.exists(fname+'_comments.csv'):
        print(fname+'_comments.csv', 'does not exists, creating a new csv file')
        append_write = 'w'

    with open(fname+'_comments.csv', append_write) as comments_file:
        comments_writer = csv.writer(comments_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        if comment_file:
            comments_writer.writerow(['publish_time', 'Video ID', 'Title', 'Comment', 'updatedAt', 'likeCount'])
        elif append_write == 'w':
            comments_writer.writerow(['publish_time', 'video_ID', 'Title', 'viewCount',
                                      'likeCount', 'dislikeCount', 'favoriteCount', 'commentCount'])
        for row in Youtube_res_list:
            # convert the tuple to a list and write to the output file
            comments_writer.writerow(list(row))


def get_videos(service, **kwargs):
    final_results = []
    results = service.search().list(**kwargs).execute()

    i = 0
    max_pages = 1
    while results and i < max_pages:
        final_results.extend(results['items'])

        # Check if another page exists
        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']
            results = service.search().list(**kwargs).execute()
            i += 1
        else:
            break

    print("Total results: {0} \nResults per page: {1}".format(results['pageInfo']['totalResults'],
                                                              results['pageInfo']['resultsPerPage']))
    print("Example output per item, snippet")
    print(results['items'][0]['snippet'].keys())
    return final_results


def search_videos_by_keyword(service, game, **kwargs):
    results = get_videos(service, **kwargs)
    final_result = []
    game_summary = []
    for item in results:
        #if item["id"]["kind"] == "youtube#video":
        publish_time = item['snippet']['publishedAt']
        title = item['snippet']['title']
        video_id = item['id']['videoId']

        stats = get_video_stat(service, part='snippet, statistics', id=video_id)

        comments = get_video_comments(service, part='snippet',
                                      #pageToken=99,
                                      maxResults=100,
                                      videoId=video_id, textFormat='plainText')
        # make a tuple consisting of the video id, title, comment and add the result to the final list
        # final_result.extend([(video_id, title, comment) for comment in comments])
        final_result.extend([[publish_time, video_id, title] + comment for comment in comments])
        game_summary.extend([[publish_time, video_id, title] + stat for stat in stats])
        print('Finished scraping', title)
    write_to_csv(final_result, game, True)
    return final_result, game_summary



if __name__ == '__main__':
    '''************ Input ************'''
    # region input
    Input_keyword = True  # if True, need to type in search keyword on terminal
    max_result_per_page = 1  # display 1-2 video in the search result, default 5 is too many
    max_comment_per_video = 30000
    second_time_download = False

    if Input_keyword:
        keyword = input('Enter a keyword: ')  # eg.'Official Call of Duty: Infinite Warfare Reveal Trailer'
    else:
        input_file = 'input_game_sh.csv'
        Game_names = pd.read_csv(input_file, sep=',', header=None, names=['Game'])
        Game_names.Game = Game_names.Game.astype(str)
        search_suffix = ' reveal trailer'  # keyword will be game name + suffix, eg. 'Call of Duty reveal trailer'

     # endregion input
    '''************ End of Input ************'''

    # When running locally, disable OAuthlib's HTTPs verification.
    # When running in production *do not* leave this option enabled.
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    service = get_authenticated_service()

    Game_comment_dict = {}
    overall_game_summary = []

    if Input_keyword:
        keyword = keyword.translate(str.maketrans('', '', string.punctuation))
        print("\nSearching  {}  now...".format(keyword))
        final_result, game_summary = search_videos_by_keyword(service, keyword, q=keyword,  order='relevance',
                                                                                maxResults=max_result_per_page,
                                                                                #regionCode='HK', eventType='completed',
                                                                                part='id,snippet', type='video')
        Game_comment_dict[keyword] = final_result
        overall_game_summary.extend(game_summary)
    else:
        # g = Game_names[0]
        for g in Game_names.Game:
            g = g.translate(str.maketrans('', '', string.punctuation))
            keyword = g + search_suffix
            print("\nSearching  {}  now...".format(keyword))

            final_result, game_summary = search_videos_by_keyword(service, g, q=keyword,  order='relevance',
                                                                                maxResults=max_result_per_page,
                                                                                #regionCode='HK', eventType='completed',
                                                                                part='id,snippet', type='video')
            Game_comment_dict[keyword] = final_result
            overall_game_summary.extend(game_summary)

    write_to_csv(overall_game_summary, 'overall', False, 'a')



