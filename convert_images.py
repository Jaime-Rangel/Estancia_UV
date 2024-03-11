import os
from os import listdir
from os.path import isfile, join
import numpy
from PIL import Image

print(os.getcwd())

localpath = os.getcwd();

path = os.path.join(localpath, "renames")

savedir = path

if not os.path.exists(path):
  os.mkdir(path)
  print("Folder %s created!" % path)
else:
  print("Folder %s already exists" % path)

onlyfiles = [f for f in listdir(localpath) if f.endswith(".jpg")]
images = numpy.empty(len(onlyfiles), dtype=object)

for n in range(0, len(onlyfiles)):
    filename = onlyfiles[n].split('.', 1)
    img = Image.open(join(localpath,onlyfiles[n]) )
    img.save(str(join(savedir, filename[0] + ".jpeg")),subsampling=0, quality=100)