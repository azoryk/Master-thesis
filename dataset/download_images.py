import flickrapi
import sys
import shutil
import urllib.request
import os

print(flickrapi.__version__)

api_key = '285a24d486541ae9086afcfefd109f07' # flickr key
api_secret = '24d0976e877a2a82' #flickr secret

NUMBER_OF_IMAGES = 1000

flickr = flickrapi.FlickrAPI(api_key, api_secret)
photos = flickr.photos.search(user_id='94232412@N00', per_page='10')
sets = flickr.photosets.getList(user_id='94232412@N00')

#
# def get_urls_for_tags(tags):
#    # photos = flickr.photos_search(tags=tags, tag_mode='all', per_page=number)
#
#     photos = flickr.walk(text=tags, tag_mode='all', tags=tags,
#                                                   extras='url_c',
#                                                   per_page=500,
#                                                   sort="relevance")
#
#     urls = []
#
#     for photo in photos:
#         try:
#             urls.append(photo.getURL(size='Large', urlType='source'))
#             #urls.append(photo.get('url_c'))
#         except:
#             continue
#     return urls
#
#
# def download_images(urls):
#
#     for url in urls:
#         file, mime = urllib.request.urlretrieve(url)
#
#         name = url.split('/')[-1]
#         print(name)
#         shutil.copy(file, './' + name)
#
#
# def main(*argv):
#     args = argv[1:]
#
#     if len(args) == 0:
#         print("You must specify at least one tag")
#         return 1
#
#     tags = [item for item in args]
#
#     urls = get_urls_for_tags(tags)
#     download_images(urls)
#
# if __name__ == '__main__':
#     sys.exit(main(*sys.argv))

'''
This program finds the urls for the mentioned tag, e.g. 'Bamberger dom'
'''

# print (sys.argv)
# if len(sys.argv) > 1:
#     print ("Reading queries from file " + sys.argv[1])
#     query_file_name = sys.argv[1] #0 is the command name.
#     file = open(query_file_name)
#     print(file.read())
# else:
#     print ("No command line arguments, reading queries from " + 'queries.txt')
#     query_file_name = 'germany_poi.txt'

query_file_name = 'germany_poi.txt'
query_file = open(query_file_name, 'r', encoding='utf-8-sig')
queris = []

for line in query_file:
   queris = queris + [line[0:len(line) - 1]]

#os.mkdir('image_tags_urls')

def get_url_for_tags(keyward):
   count_url = 0
   print("tags to search: " )
   print(keyward)
   for tag in keyward:
       print(tag)
       print("Images by tag '" + tag + "' are searching...")
       photos = flickr.walk(text=tag,
                        tag_mode='all',
                        tags=tag,
                        extras='url_c',
                        per_page=500,
                        sort="relevance",
                        )
       file = open(tag + "_photos_url.txt", "w", encoding='utf-8-sig')

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
       # file.close()
       # shutil.move(file, 'image_tags_urls/')
       urls = set(urls)
       print(str(len(urls)) + ' urls were found' )
   return urls

def download_images(urls):
   os.mkdir('cologne_cathedral')
   for url in urls:
       # file, mime = urllib.request.urlretrieve(url)
       folder_name = 'cologne_cathedral/'
       name = url.split('/')[-1]
       f = open(folder_name + name, 'wb')
       f.write(urllib.request.urlopen(url).read())
       f.close()
       try:
           shutil.move(name, 'cologne_cathedral/')
       except:
           continue

       # print(name)
       # shutil.copy(file, './' + name)




#urls= get_url_for_tags(queris)


urls_file_name = 'cologne_cathedral_photos_url'
urls_file = open(urls_file_name, 'r', encoding='utf-8-sig')
urls = []

for line in urls_file:
   urls = urls + [line[0:len(line) - 1]]



download_images(urls)
