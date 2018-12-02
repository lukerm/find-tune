## Imports ##

import re
import json


## Constants ## 

# Take more than one per category, just in case it fails
N_TRACKS_PER_CAT = 2


## Functions ##

def find_youtube_links(category, ontology):
    for o in ontology:
        if o['name'] == category.strip():
            return o['positive_examples']

    return []


def parse_youtube_url(url):
    if 'http' not in url:
        url = 'https://' + url

    yt_id = url.split('/')[3].split('?')[0]
    s_start = re.search('start=([0-9]+)', url).group(1)
    s_end = re.search('end=([0-9]+)', url).group(1)

    return (url, yt_id, s_start, s_end)	


## Main ## 

# Load the ontology, to seek examples
with open('data/ontology.json', 'r') as f:
    ontology = json.load(f)

# Open the list of categories we think are relevant counter-examples (non-target noises)
with open('data/non-target_categories.txt', 'r') as f:
    sound_cats = f.readlines()

# Process each of the categories, writing and parsing the first URLs
link_details = {}
with open('data/dl_youtube_links.txt', 'w') as f: 

    for cat in sound_cats:
        cat = cat.strip()
        f.write('# %s\n' % cat)
        my_cat_links = find_youtube_links(cat, ontology)[:N_TRACKS_PER_CAT]
        for url in my_cat_links:
            f.write(url) 
            f.write('\n')
        
        link_details[cat] = [parse_youtube_url(url) for url in my_cat_links]

# Save the parsed URLs to file
with open('data/link_details.json', 'w') as j:
     json.dump(link_details, j)
