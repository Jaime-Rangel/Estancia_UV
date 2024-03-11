import os
from os import listdir
from os.path import isfile, join
import numpy
from PIL import Image

print(os.getcwd())
localpath = os.getcwd();
path = os.path.join(localpath, "resize")
savedir = path

def change_imgsize(factor_escala):
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

        ancho, alto = img.size

        nuevo_ancho = int(ancho * factor_escala)
        nuevo_alto = int(alto * factor_escala)

        img = img.resize((nuevo_ancho, nuevo_alto))
        img.save(str(join(savedir, filename[0] + ".jpg")),subsampling=0, quality=100)

factor_escala = 0.33

change_imgsize(factor_escala)