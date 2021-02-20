



import os

path = "/home/ubuntu/Documents/cm/contest_frame/"

def rename_file():
    cnt = 0
    for filename in os.listdir("./contest_frame"):
        c = str(cnt)
        
        newfilename = "frame" + c + ".jpg"
        cnt += 1
        
        

        os.rename(path + filename, path + str(newfilename))

if __name__ == "__main__":

    
    rename_file()

