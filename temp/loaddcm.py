import dicom
import os
import numpy as np
from PIL import Image

img_path = "/nfs/bigbrain/vhnguyen/projects/radiomics/a/2/000000.dcm";
lst_slice = [];

cInfo = dicom.read_file(img_path);

rois = cInfo.ROIContourSequence;
roi0 = cInfo.ROIContours[0];

a = roi0.Contours[0].ContourData;
print type(a);
print len(a);
