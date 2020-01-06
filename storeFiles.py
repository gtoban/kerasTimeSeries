import subprocess
import datetime
import sys
import os

def main():
    if (len(sys.argv) < 2):
        print("id number required")
        exit(0)

    idNumI = int(sys.argv[1])
    d = datetime.date.today()
    idDate = f"{idNumI:05d}.{d.year}{d.month:02d}{d.day:02d}."
    model = idDate + "fileModel.csv"
    result = idDate + "fileResult.csv"
    params = idDate + "fileTrainTestParams.txt"
    status = idDate + "status.txt"
    
    if (os.stat("fileModel.csv").st_size > 0):
        p = subprocess.Popen(["cp", "fileModel.csv", model])
    if (os.stat("fileResult.csv").st_size > 0):
        p = subprocess.Popen(["cp", "fileResult.csv", result])
    if (os.stat("fileTrainTestParams.txt").st_size > 0):
        p = subprocess.Popen(["cp", "fileTrainTestParams.txt", params])
    if (os.stat("status.txt").st_size > 0):
        p = subprocess.Popen(["cp", "status.txt", status])
    

main()

