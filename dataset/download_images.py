import flickrapi
import sys
import shutil
import urllib.request
import os

print(flickrapi.__version__)

api_key = '285a24d486541ae9086afcfefd109f07' # flickr key
api_secret = '24d0976e877a2a82' #flickr secret

flickr = flickrapi.FlickrAPI(api_key, api_secret)
photos = flickr.photos.search(user_id='94232412@N00', per_page='10')
sets = flickr.photosets.getList(user_id='94232412@N00')

'''
This program finds the urls for the mentioned tag, e.g. 'Bamberger dom'
'''

def get_url_for_tags(keyward):
   os.mkdir('image_tags_urls')
   count_url = 0
   print("tags to search: " )
   print(keyward)
   for tag in keyward:
       print(tag)
       print("Images by tag '" + tag + "' are searching...")
       photos = flickr.walk(api_key=api_key,
                                        privacy_filter="1",
                                        media="photos",
                                        per_page=500,
                                        tag_mode = 'all',
                                        tags = tag,
                                        text=tag,
                                        accuracy="16",
                                        extras = 'url_c',
                                        sort="relevance"

                         )
       file_path = tag + "_photos_url.txt"
       file = open(file_path, "w", encoding='utf-8-sig')

       urls = []
       for photo in photos:
           try:
               url=photo.get('url_c')
               if url not in urls:
                   urls.append(url)
                   file.write(url + '\n')
                   count_url += 1

           except Exception as e:
               continue
       file.close()

       shutil.move(file_path, 'image_tags_urls/')
       urls = set(urls)
       print(str(len(urls)) + ' urls were found' )
   return urls


def download_images(path):  # argument - path to the folder
    query_file_name = 'germany_poi.txt'

    query_file = open(query_file_name, 'r', encoding='utf-8-sig')
    queris = []
    print('reading query file')
    for line in query_file:
        queris = queris + [line[0:len(line) - 1]]
    url_list = []
    print('tags to search: ')
    print(queris)
    for file in queris:
        url_file = open(path +file + '_photos_url.txt', 'r', encoding='utf-8-sig')
        print('reading "' + file + '_photos_url.txt"...' )
        for url1 in url_file:
            url_list = url_list + [url1[0:len(url1) - 1]]
        folder_path =  file + '_images'
        print('creating folder "' + file + '_images"')
        os.mkdir(folder_path)
        print('downloading "' + file  + '" images...')
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

