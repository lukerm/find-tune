## Imports ##

import os
import re
import json

from definitions import DATA_DIR


## Constants ##

# Take more than one per category, just in case it fails
N_TRACKS_PER_CAT = 2


## Functions ##

def prepend_protocol(url):
    if 'http' not in url:
        return 'https://' + url
    else:
        return url


def find_youtube_links(category, ontology):
    for o in ontology:
        if o['name'] == category.strip():
            return o['positive_examples']

    return []


def parse_youtube_url(url):
    url   = prepend_protocol(url)
    yt_id = url.split('/')[3].split('?')[0]
    s_start = re.search('start=([0-9]+)', url).group(1)
    s_end = re.search('end=([0-9]+)', url).group(1)

    return (url, yt_id, s_start, s_end)	


## Main ## 

# Load the ontology, to seek examples
with open(os.path.join(DATA_DIR, 'ontology.json'), 'r') as f:
    ontology = json.load(f)

# Open the list of categories we think are relevant counter-examples (non-target noises)
with open(os.path.join(DATA_DIR, 'non-target_categories.txt'), 'r') as f:
    sound_cats = f.readlines()

# Process each of the categories, writing and parsing the first URLs
link_details = {}
n_cats  = 0
n_links = 0
link_fname = 'data/dl_youtube_links.txt'
with open(link_fname, 'w') as f: 
    print('Saving links to: %s' % link_fname)
    for cat in sound_cats:
        cat = cat.strip()
        f.write('# %s\n' % cat)
        my_cat_links = find_youtube_links(cat, ontology)[:N_TRACKS_PER_CAT]

        # Verbose 
        n_new_links = len(my_cat_links)
        n_links += n_new_links
        if n_new_links > 0:
            n_cats += 1
 
        for url in my_cat_links:
            f.write(prepend_protocol(url))
            f.write('\n')
        
        link_details[cat] = [parse_youtube_url(url) for url in my_cat_links]


# Save the parsed URLs to file
link_detail_fname = os.path.join(DATA_DIR, 'link_details.json')
print('Saving link details to: %s' % link_detail_fname)
with open(link_detail_fname, 'w') as j:
     json.dump(link_details, j)

print('Found %d links from %d non-empty categories' % (n_links, n_cats))
print('')
