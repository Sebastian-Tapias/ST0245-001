import os
import pandas as pd
#location of the folder containing the csv files
location= "/Users/valeriacardona/OneDrive - Universidad EAFIT/2do Semestre/DATOS Y ALGORITMOS/Proyecto/Entrega 1/Codigo/datasets/csv"+"/"
os.chdir(location)     # change the current working directory to specified path(to avoid errors)
filesArray = sorted(os.listdir(os.getcwd()))
print(os.getcwd())
#os.remove(location+".DS_Store")# -->ACTIVATE THIS IF A .DS_Store FILE IS CREATED 
print(filesArray)

def uploadFiles (path, folder):
        store= []
        print("Loading files from : "+folder)
        for file in os.listdir(path+str(folder)):
            store.append(pd.read_csv(path+folder+"/"+file))
            print("Loading File "+ str(len(store)))#It's just to see how many files have been uploaded.
        return store 

print(uploadFiles(location, filesArray[-1]))#  The index indicates the csv folder I want to read
#Write index number or folder's name