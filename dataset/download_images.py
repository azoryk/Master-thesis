import flickrapi
import sys
import shutil
import urllib.request
import os
from datetime import datetime

'''
This program finds the urls for the mentioned tag, e.g. 'Bamberger dom'
'''

api_key = 'e30dd4e80f5878e23cb2766cebe3a3bd' # flickr key
api_secret = '47f19e7786d0f018' #flickr secret

flickr = flickrapi.FlickrAPI(api_key, api_secret, cache=True)


def get_url_for_tags(keyward):
    # the directory where all output files will be saved
    os.mkdir('image_tags_urls')


    count = 0


    print("tags to search: " )
    print(keyward)
    for tag in keyward:
        #create output file with all the urls
        urls = []
        count_url = 0
        file_name = tag + "_photos_url.txt"
        file = open(file_name, "a", encoding='utf-8-sig')

        #timestamps to download all the images for the last 15 years
        mintime = 1009843200  # 01/01/2002
        maxtime = 1041379200  # 01/01/2003
        endtime = 1494806400  # nowadays 15.05.2017
        #timeskip = 31536000  # one year skipping
        timeskip = 15768000 # half year skipping
        desired_photos = 3000
        print("Images by tag '" + tag + "' are searching...")
        while (maxtime < endtime and count_url < 15000):
            try:
                print("Searching images from " + str(datetime.fromtimestamp(mintime)))
                print("till " + str(datetime.fromtimestamp(maxtime)))

                photos = flickr.walk(api_key=api_key,
                                       #privacy_filter="1",
                                       #media="photos",
                                       per_page=250,
                                       tag_mode = 'all',
                                       #tags = tag,
                                       text=tag,
                                       #accuracy="6",
                                       extras = 'url_c',
                                       sort="relevance",
                                       #min_upload_date=str(mintime),
                                       #max_upload_date=str(maxtime),
                                       min_taken_date=str(mintime),
                                       max_taken_date=str(maxtime)

                        )
                #total_images = photos.photos[0]['total']
                #print(total_images)
                #all the urls will be saved in this .txt file
                # if (len(photos) > desired_photos):
                #     print('too many photos in block, reducing maxtime')
                #     maxtime = (mintime + maxtime) / 2


                for photo in photos:
                    try:
                        url=photo.get('url_c')
                        if url not in urls:
                            urls.append(url)
                           #with open(file_name, "a") as myfile:
                            #myfile.write("url + '\n'")
                            #file.seek(0)
                            file.write(url + '\n')
                            count_url += 1
                    except Exception as e:
                         continue


                #urls = set(urls)
                print(str(count_url) + ' unique images of ' + tag + ' were found')
                mintime = maxtime
                maxtime = maxtime + timeskip
                print('the end of phase')
            except:
                continue
        file.close()
        shutil.move(file_name, 'image_tags_urls/')


            #return urls


def download_images(path):  # argument - path to the folder
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
       folder_path = 'dataset_final/' + file + '_images'
       print('Creating folder "' + file + '_images"')
       os.mkdir(folder_path)
       print('Downloading "' + file  + '" images...')
       print('It will be ' + str(len(url_list)) + ' images downloaded')
       for url in url_list:
           name = url.split('/')[-1]
           f = open(folder_path + '/' + name, 'wb')
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

   query_file_name = 'germany_poi.txt'
   query_file = open(query_file_name, 'r', encoding='utf-8-sig')
   queris = []

   for line in query_file:
       queris = queris + [line[0:len(line) - 1]]

   get_url_for_tags(queris)
   path = 'image_tags_urls/'
   download_images(path)

if __name__ == '__main__':
   sys.exit(main(*sys.argv))
