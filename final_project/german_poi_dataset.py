'''
Image querying and downloading script written by Andriy Zoryk
hinted from "Code for finding and downloading images on Flickr"
http://graphics.cs.cmu.edu/projects/im2gps/flickr_code.html  and
https://stackoverflow.com
'''
import flickrapi
import sys
import shutil
import urllib.request
import os
from datetime import datetime


#To use Flickr API we need to use flickr auth information:
#flickr api key and secret, which could be get on 
#https://www.flickr.com/services/api/ 

api_key = '285a24d486541ae9086afcfefd109f07' # flickr key
api_secret = '24d0976e877a2a82' #flickr secret

#make a new FlickrAPI inctance
flickr = flickrapi.FlickrAPI(api_key, api_secret, cache=True)

#this method obtain the urls of the images by the tag and saves them to 
#.txt file with the name of tag
def get_url_for_tags(keyward):
   # the directory where all output files will be saved
   if not os.path.isdir('image_tags_urls'):
       os.mkdir('image_tags_urls')
   print("tags to search: " )
   print(keyward)
   for tag in keyward:
       #output file with all the urls
       file_name = tag + "_photos_url.txt"
       file = open(file_name, "a", encoding='utf-8-sig')
       urls = []
       count_url = 0

#flickr api allows only to obtain 4K image urls at once, that's why we need
#to search the batches of images not more than 4K

       #timestamps to download all the images for the last 15 years
       mintime = 1009843200  # 01/01/2002
       maxtime = 1041379200  # 01/01/2003
       endtime = 1494806400  # nowadays 15.05.2017
       #timeskip = 31536000  # one year skipping
       timeskip = 15768000 # half year skipping
       print("Images by tag '" + tag + "' are searching...")
       #we will search the images from the very beginning till their total number will be 11K
       while (maxtime < endtime and len(urls)<11000):
           try:
               print("Searching images from " + str(datetime.fromtimestamp(mintime)))
               print("Searching is till " + str(datetime.fromtimestamp(maxtime)))

               photos = flickr.walk(api_key=api_key,
                                      #privacy_filter="1",
                                      #media="photos",
                                      per_page=250,
                                      tag_mode = 'all',
                                      tags = tag,
                                      text=tag,
                                      #accuracy="6",
                                      extras = 'url_c',
                                      #sort="relevance",
                                      min_taken_date=str(mintime),
                                      max_taken_date=str(maxtime)
                                      )
               for photo in photos:
                    try:
                        url=photo.get('url_c')
                        if url not in urls:
                            urls.append(url)
                            file.write(url + '\n')
                            count_url += 1

                    except Exception as e:
                        continue
               mintime = maxtime
               maxtime = maxtime + timeskip
               print('total amount of urls: ' + str(len(urls)))
               print('the end of phase')
           except:
               continue
       file.close()
       shutil.move(file_name, 'image_tags_urls/')


#this method reads the image urls from the .txt file and dowloads the images to the output location 
def download_images(path):  # argument - path to the folder with urls files

  if not os.path.isdir('dataset_final'):
      os.mkdir('dataset_final')
  query_file_name = 'germany_poi.txt'
  query_file = open(query_file_name, 'r', encoding='utf-8-sig')
  queris = []
  print('Reading query file...')
  for line in query_file:
      queris = queris + [line[0:len(line) - 1]]
  print('Tags to search: ')
  print(queris)
  for file in queris:
      url_list = []
      url_file = open(path +file + '_photos_url.txt', 'r', encoding='utf-8-sig')
      print('Reading "' + file + '_photos_url.txt"...' )
      for url1 in url_file:
          url_list = url_list + [url1[0:len(url1) - 1]]
      url_list = set(url_list)
      folder_path =  'dataset_final/' + file + '_images'
      print('Creating folder "' + file + '_images"')
      os.mkdir(folder_path)
      print('Downloading "' + file  + '" images...')
      print('It will be ' + str(len(url_list)) + ' downloaded')
      for url in url_list:
          name = url.split('/')[-1]
          f = open(folder_path + '/' + name, 'wb')
          #reads the image and copies it to the local file 
          f.write(urllib.request.urlopen(url).read())
          print(name + ' was downloaded')
          f.close()
          try:
              shutil.move(name, folder_path)

          except:
              continue

def main(*argv):
  print (sys.argv)
  if len(sys.argv) > 1:
      print ("Reading queries from file " + sys.argv[1])
      query_file_name = sys.argv[1] #the file 'german_poi.txt' should be added in cmd.
  else:
      print ("No command line arguments, reading queries from " + 'germany_poi.txt')

#the .txt file where the list of all landmarks was saved
  query_file_name = 'germany_poi.txt'
  query_file = open(query_file_name, 'r', encoding='utf-8-sig')
  queris = []

#we've got the list of landmarks from the .txt file 
  for line in query_file:
      queris = queris + [line[0:len(line) - 1]]

  get_url_for_tags(queris)

#provide the path to your .txt file with the urls 
  path = 'christkindl/'
  download_images(path)

if __name__ == '__main__':
  sys.exit(main(*sys.argv))

