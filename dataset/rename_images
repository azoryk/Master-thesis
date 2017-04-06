#script to rename images in the dataset directories 

import os
directoryname="C:\\Users\\azoryk\\Desktop\\dataset\\3.venedig"
lijstmetfiles = os.listdir(directoryname)
print(lijstmetfiles)
for i in range(len(lijstmetfiles)):

    os.rename(
        os.path.join(directoryname, lijstmetfiles[i]),
        os.path.join(directoryname, "venedig-"+str(i).zfill() + '.jpg')
        )
