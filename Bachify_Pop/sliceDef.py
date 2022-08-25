# stuff you need to import for the definition to work

import os
from pydub import AudioSegment
from pydub.utils import make_chunks

# here's the definition
def sliceFromList(file, list):
    listLen=len(list)
# get the number of pairs in the list (divide it by 2), however, now you need to make a pair from each so it's list len minus 1
    noOfPairsInList=listLen-1
# turn that division into an int
    pairCount=int(noOfPairsInList)

# file name you want to chop
    fileName=(file)
    nameBase=os.path.splitext(fileName)[0]
    print(nameBase)
# dump the audio into pydub's audioSegment tool. This means we can then call myaudio and make a slice
    myaudio = AudioSegment.from_file(fileName, "wav")
# loop through the pairs in the list, so this will run for the number of pairs you have in the list (pairCount, defined above)
    for i in range(pairCount):
        fileNo=i # quickly catch the int so we can use it in the fileName

    #this clunkily gets pairs from the list by knowing which iteration of the loop we are in (i) and multiplying by 2 once it's been through the first pair
    #    if i==0:
    #        i=i
    #    else:
    #        i=(i*2)
# create readable name parts of start and end times for filename
        name_chunk_start=str(round(list[i]))
        name_chunk_end=str(round(list[i+1]))
# pydub calculates in millisec so need to * 1000
        chunk_start = list[i]*1000
        chunk_end = list[i+1]*1000
    # create a chunk from the file called myaudio
        chunk=myaudio[chunk_start:chunk_end]
    # create a filename to export to
        chunk_name = os.path.join(nameBase + "_" + name_chunk_start + "_" + name_chunk_end + ".wav")
    # export the chunk as a wav file
        chunk.export(chunk_name, format="wav")

# --- alphebet iteration ---
# this needs to be before the chunk names
import string
string.ascii_uppercase
