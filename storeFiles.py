import subprocess
import datetime
import sys

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

    p = subprocess.Popen(["cp", "fileModel.csv", model])
    p = subprocess.Popen(["cp", "fileResult.csv", result])
    p = subprocess.Popen(["cp", "fileTrainTestParams.txt", params])
    

main()

