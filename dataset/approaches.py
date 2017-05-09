import urllib.request
import flickrapi

api_key = '285a24d486541ae9086afcfefd109f07'
api_secret = '24d0976e877a2a82'

flickr = flickrapi.FlickrAPI(api_key, api_secret, cache=True)


def flickr_walk(keyward):
    count = 0
    photos = flickr.walk(text=keyward,
                         tag_mode='all',
                         tags=keyward,
                         extras='url_c',
                         per_page=100)

    for photo in photos:
        try:
            url = photo.get('url_c')
            urllib.request.urlretrieve(url, path + '\\' + str(count) + ".jpg")
        except Exception as e:
            print('failed to download image')


flickr_walk('Bamberger Dom')


# this is slow
def get_urls_for_tags(tags, number):
    photos = flickr.photos_search(tags=tags, tag_mode='all', per_page=number)
    urls = []
    for photo in photos:
        try:
            urls.append(photo.getURL(size='Large', urlType='source'))
        except:
            continue
    return urls


def download_images(urls):
    for url in urls:
        file, mime = urllib.urlretrieve(url)
        name = url.split('/')[-1]
        print(name)
        shutil.copy(file, './' + name)


def main(*argv):
    args = argv[1:]
    if len(args) == 0:
        print("You must specify at least one tag")
        return 1

    tags = [item for item in args]

    urls = get_urls_for_tags(tags, NUMBER_OF_IMAGES)
    download_images(urls)


if __name__ == '__main__':
    sys.exit(main(*sys.argv))