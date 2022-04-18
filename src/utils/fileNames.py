import os

def is_audio(path):
    """Check allowed audio formats"""

    fileName = os.path.split(path)[-1]        
    extention = fileName.split('.')[-1]
    extention = extention.lower()
    
    return extention in ["mp3","flac","wav"]

def getListOfFiles(path):
    """ 
        Create a list of file and sub directories              
    """

    listOfFile = os.listdir(path)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(path, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            if (is_audio(fullPath)):
                allFiles.append(fullPath)
            
    return allFiles


def getFileNamesWithDirs(directory):

    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    print(subdirs)
    f = open(r"I:\\Downloads\\program_test\\bard\\dataset\\all_labels.txt",'w')
    for elems in subdirs:
        files = getListOfFiles(os.path.join(directory, elems))
        for file in files:           
            f.write(os.path.basename(file) + "\t\t" + elems+"\n")
            print(os.path.basename(file) + "\t\t" + elems+"\n")

    f.close()

getFileNamesWithDirs(r"I:\\Downloads\\program_test\\bard\\dataset")
