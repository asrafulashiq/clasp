import json
from pathlib import Path
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

annFile = './bin_detection.json'
ROOT = Path(os.environ['HOME'] + '/dataset/clasp_data')

coco = COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))


catIds = coco.getCatIds(catNms=['binEMPTY'])
imgIds = coco.getImgIds(catIds=catIds)
#imgIds = coco.getImgIds(imgIds = [324158])
while True:
	plt.close('all')
	img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

	imfile = ROOT / img['file_name']

	I = cv2.imread(str(imfile))

	# load and display instance annotations
	print(imfile)
	annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
	anns = coco.loadAnns(annIds)

	for ann in anns:
	    bb = ann['bbox']
	    bb[2] = bb[0] + bb[2] -1
	    bb[3] = bb[1] + bb[3] -1
	    bb = tuple([int(i) for i in bb])
	    cv2.rectangle(I, bb[:2], bb[2:], (0, 255, 0), 5)

	plt.imshow(I[:,:,::-1])
	plt.axis('off')
	plt.show()