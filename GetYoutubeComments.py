#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GetYoutubeComments.py
@author: Sherry He

This file uses Google API client to call Youtube Data API v3,
and extract comments from search results
To run this code, first set up Google api client and OAUTH2
Documentation: 
https://developers.google.com/youtube/v3/getting-started
"""

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
        try:
            #if item["id"]["kind"] == "youtube#video":
            publish_time = item['snippet']['publishedAt']
            title = item['snippet']['title']
            video_id = item['id']['videoId']

            stats = get_video_stat(service, part='snippet, statistics', id=video_id)


            comments = get_video_comments(service, part='snippet',
                                         maxResults=100,
                                         videoId=video_id, textFormat='plainText')
    
            #make a tuple consisting of the video id, title, comment and add the result to the final list
            #final_result.extend([(video_id, title, comment) for comment in comments])
            
            final_result.extend([[publish_time, video_id, title] + comment for comment in comments])

            game_summary.extend([[publish_time, video_id, title] + stat for stat in stats])
            print('Finished scraping', title)
        except Exception as e:
            print (e)
            write_to_csv(overall_game_summary, 'overall', False, 'a')

    write_to_csv(final_result, game, True)
    return final_result, game_summary



if __name__ == '__main__':
    '''************ Input ************'''
    # region input
    Input_keyword = True  # if True, need to type in search keyword on terminal
    max_result_per_page = 1  # display 1-2 video in the search result, default 5 is too many
    max_comment_per_video = 60000
    second_time_download = False

    if Input_keyword:
        keyword = input('Enter Youtube search keyword: ')  # eg.'Official Call of Duty: Infinite Warfare Reveal Trailer'
    else:
        input_file = 'input_movie_sh.csv' #'input_game_sh.csv'
        Game_names = pd.read_csv(input_file, sep=',', header=None, names=['Game'])
        Game_names.Game = Game_names.Game.astype(str)
        search_suffix =  ' official trailer'#' reveal trailer'  # keyword will be game name + suffix, eg. 'Call of Duty reveal trailer'

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

###### End of Code #####

'''  movie list - only my part
Paramount = ["Bumblebee","Instant Family","Overlord", "Nobody's Fool",
"Mission: Impossible – Fallout", "Action Point","Book Club","A Quiet Place",
"Sherlock Gnomes","Annihilation",
"The Cloverfield Paradox", "Downsizing","Daddy's Home 2","Suburbicon", 
"Same Kind of Different as Me", "Mother!","Tulip Fever",
"An Inconvenient Sequel: Truth to Power", "Transformers: The Last Knight",
"Baywatch","Ghost in the Shell","Rings","XXX: Return of Xander Cage", 
"Monster Trucks","Silence","Fences","Office Christmas Party", "Allied", "Arrival", 
"Jack Reacher: Never Go Back",
"Goat", "The Intervention","Ben-Hur","Florence Foster Jenkins","The Little Prince",
"Star Trek Beyond","Approaching the Unknown","Teenage Mutant Ninja Turtles: Out of the Shadows",
"Everybody Wants Some!!","10 Cloverfield Lane",
"Whiskey Tango Foxtrot","Zoolander 2","13 Hours: The Secret Soldiers of Benghazi",
"Anomalisa","Daddy's Home","The Big Short",
"Scouts Guide to the Zombie Apocalypse",
"Paranormal Activity: The Ghost Dimension",
"Captive","Mission: Impossible – Rogue Nation", 
"Drunk Wedding","Area 51",

"Hot Tub Time Machine 2",
"The SpongeBob Movie: Sponge Out of Water","Project Almanac",
"Selma","The Gambler","Top Five","Interstellar",
"Men, Women & Children","Teenage Mutant Ninja Turtles","Hercules",
"Transformers: Age of Extinction","Noah",
"Katy Perry: Part of Me",
"Madagascar 3: Europe's Most Wanted",
"The Dictator",
"Titanic 3-D",
"A Thousand Words",
"The Devil Inside",
"The Adventures of Tintin: Secret of the Unicorn",
"Young Adult",
"Hugo",
"Puss in Boots",
"Paranormal Activity 3",
"Footloose",
"Cowboys & Aliens",
"Transformers: Dark of the Moon",
"Super 8",
"Kung Fu Panda 2",
"Rango",
"Justin Bieber: Never Say Never",
"No Strings Attached",
"True Grit",
"The Fighter",
"Morning Glory",
"Megamind",
"Paranormal Activity 2",
"Jackass 3D",
"Dinner for Schmucks",
"The Last Airbender",
"Shrek Forever After",
"How to Train Your Dragon",
"Shutter Island",

"Jack Ryan: Shadow Recruit",
"Paranormal Activity: The Marked Ones",
"Labor Day",
"The Wolf of Wall Street",
"Anchorman 2: The Legend Continues",
"Jackass Presents: Bad Grandpa",
"World War Z",
"Star Trek Into Darkness",
"Pain & Gain",
"Hansel & Gretel: Witch Hunters",
"Jack Reacher",
"The Guilt Trip","Rise of the Guardians",
"Flight",
"Fun Size",
"Paranormal Activity 4"]
'''

'''
Bridget Jones's Baby - Official Trailer
Ouija: Origin of Evil - Official Trailer
Almost Christmas - Official Trailer
Fast & Furious 6 Official Trailer #1
The Fate of the Furious - Official Trailer

The Mummy - Official Trailer #2
American Made - Official Trailer
Pitch Perfect 3 - Official Trailer 2
Pacific Rim Uprising - Official Trailer 2 [HD]
Johnny English Strikes Again - Official Trailer
Green Book - Official Trailer

Paramount
Kung Fu Panda 2 | Official Teaser Trailer

universal
Minions - Official Trailer (HD) - Illumination
Fifty Shades Of Grey - Official Trailer
The Secret Life Of Pets 2 - Official Trailer
The Secret Life Of Pets - Official Teaser Trailer (HD) - Illumination
Lucy - Trailer
Fast Five - Teaser Trailer
Snow White and the Huntsman Official Movie Trailer
Dr. Seuss' The Lorax (2012) EXCLUSIVE Trailer - HD Movie
Bridesmaids - Trailer

Sony
The Interview Final Trailer - Meet Kim Jong-Un
FURY - Official Trailer sony
SEX TAPE MOVIE - Official Red Band Trailer
SMURFS 2 (3D) - Official Trailer

warner bro
The LEGO Batman Movie - Batcave Teaser Trailer
ISN'T IT ROMANTIC - Official Trailer
Final Destination 5 - Trailer

JUSTICE LEAGUE - Official Trailer 1
THE 15:17 TO PARIS - Official Trailer [HD]
TOMB RAIDER - Official Trailer #1
READY PLAYER ONE - Official Trailer 1 [HD]
RAMPAGE - OFFICIAL TRAILER 1 [HD]
LIFE OF THE PARTY - Official Trailer 1

OCEAN'S 8 - Official 1st Trailer
TAG - Official Trailer 1
Teen Titans GO! To The Movies - Official Trailer 1
CRAZY RICH ASIANS - Official Trailer
THE NUN - Official Teaser Trailer [HD]
SMALLFOOT - Official Final Trailer [HD]
A STAR IS BORN - Official Trailer 1
THE MULE - Official Trailer

Aquaman - Official Trailer 1 - Now Playing In Theaters
TAKEN 3 | Official Trailer [HD] | 20th Century FOX
Kingsman: The Secret Service | Official Trailer 2 [HD] | 20th Century FOX
The Longest Ride | Official Trailer [HD] | 20th Century FOX
Spy | Official Trailer 2 [HD] | 20th Century FOX

Paper Towns | Official Trailer [HD] | 20th Century FOX
Fantastic Four | Official Trailer [HD] | 20th Century FOX
Hitman: Agent 47 | Official Trailer 2 [HD] | 20th Century FOX
Maze Runner: The Scorch Trials | Official Trailer [HD] | 20th Century FOX
The Martian | Teaser Trailer [HD] | 20th Century FOX

The Peanuts Movie | Official Trailer [HD] | Fox Family Entertainment
Victor Frankenstein | Official Trailer [HD] | 20th Century FOX
Alvin and the Chipmunks: The Road Chip | Official Trailer [HD] | Fox Family Entertainment
JOY | Teaser Trailer [HD] | 20th Century FOX
The Revenant | Official Teaser Trailer [HD] | 20th Century FOX
Independence Day: Resurgence | Official Trailer [HD] | 20th Century FOX
Mike and Dave Need Wedding Dates | Official Trailer [HD] | 20th Century FOX

Ice Age: Collision Course | Official Trailer #2 | 2016
Morgan | Official Trailer [HD] | 20th Century FOX
Miss Peregrine's Home for Peculiar Children | Official Trailer [HD] | 20th Century FOX
Keeping Up With the Joneses | Official Trailer [HD] | 20th Century FOX
Why Him? | Official Redband HD Trailer #1 | 2016
Hidden Figures | Teaser Trailer [HD] | 20th Century FOX

Logan | Official Trailer [HD] | 20th Century FOX
Snatched | Red Band Trailer [HD] | 20th Century FOX
Diary of a Wimpy Kid: The Long Haul | Official Trailer [HD] | Fox Family Entertainment
War for the Planet of the Apes | Official Trailer [HD] | 20th Century FOX
Kingsman: The Golden Circle | Official Trailer [HD] | 20th Century FOX
The Mountain Between Us | Official HD Trailer #1 | 2017
Ferdinand | Official Trailer [HD] | Fox Family Entertainment

The Post | Official Trailer [HD] | 20th Century FOX
Maze Runner: The Death Cure
Red Sparrow | Official Trailer [HD] | 20th Century FOX
Love, Simon | Official Trailer [HD] | 20th Century FOX
The Darkest Minds | Official Trailer [HD] | 20th Century FOX
The Predator | Official Trailer [HD] | 20th Century FOX

Bad Times at the El Royale | Official Trailer [HD] | 20th Century FOX
Bohemian Rhapsody | Official Trailer [HD] | 20th Century FOX
'''
