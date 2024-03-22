from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import os

def getText(path):
    with open(path, "r", encoding="utf-8") as file:
        html = file.read()

    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


CAPTIVES = "bdfghjklpty"

HI = "ldgptyldpytttt"

def findCaptives(text):
    
    captives = "bdfghjklpty "
    outtext = []

    for t in text:
        if t in captives:
            outtext.append(t)

    print(''.join(outtext))
    return outtext

# Einstein's Theories of Relativity and Gravitation.63372.html

filelist = []
OUTPATH = "outlines.txt"

with open("utils/data/html_only.txt", "r", encoding="utf-8") as filelist:
    files = [line.rstrip() for line in filelist]

if os.path.exists(OUTPATH):
    os.remove(OUTPATH)

with open(OUTPATH, "x", encoding="utf-8") as outlines:
    for file in files:
        if file.endswith(".html"):
            # filelist.append(line)

            text = getText(f"utils/data/gutout/{file}")

            g = findCaptives(text)

            matchstr = f"[^ {CAPTIVES}]*l[^{CAPTIVES}]*d[^{CAPTIVES}]*g[^{CAPTIVES}]*p[^{CAPTIVES}]*"


            for x in re.finditer(matchstr,text,re.MULTILINE):
                outline = f"File: {file}, Start: {x.start(0)}, End: {x.end(0)}, Text: {x.group()}"
                outlines.write(outline + "\n")
                print(outline)



