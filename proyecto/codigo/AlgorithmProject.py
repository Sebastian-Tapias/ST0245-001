from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

url_sick = "https://github.com/mauriciotoro/ST0245-Eafit/tree/master/proyecto/datasets/csv/enfermo_csv"
url_healthy = "https://github.com/mauriciotoro/ST0245-Eafit/tree/master/proyecto/datasets/csv/sano_csv"

def getFiles(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    web = soup.find_all('a', class_='js-navigation-open Link--primary')

    all_links = list()
    all_files = list()
    for i in web:
        all_links.append(i.get('href'))
    
    for j in all_links:
        index = len(j)-1
        while j[index] != '/':
            index -= 1
        all_files.append(j[index+1:])
    
    return all_files

def fillWithCSVSick(files_sick):
    array = np.empty([len(files_sick)], dtype=list)
    for i in range(len(files_sick)):
        url = "https://raw.githubusercontent.com/mauriciotoro/ST0245-Eafit/master/proyecto/datasets/csv/enfermo_csv/"+files_sick[i]
        data = pd.read_csv(url)
        data = shapeIt(data)
        data = data.to_numpy()
        array[i] = data
    
    return array

def fillWithCSVHealthy(files_healthy):
    array = np.empty([len(files_healthy)], dtype=list)
    for i in range(len(files_healthy)):
        url = "https://raw.githubusercontent.com/mauriciotoro/ST0245-Eafit/master/proyecto/datasets/csv/sano_csv/"+files_healthy[i]
        data = pd.read_csv(url)
        data = shapeIt(data)
        data = data.to_numpy()
        array[i] = data
    
    return array

def shapeIt(dataframe):
    if dataframe.shape[1] % 2 != 0:
        dataframe = dataframe.drop(columns=[f"{dataframe.columns[-1]}"])
        
    if dataframe.shape[0] % 2 != 0:
        dataframe = dataframe.drop([dataframe.shape[0]-1], axis=0)
    
    return dataframe

def compress(array):
    new_array = np.empty([int(array.shape[0]/2), int(array.shape[1]/2)], dtype = int)
    for i in range(0, array.shape[0], 2):
        for j in range(0, array.shape[1], 2):
            new_array[int(i/2)][int(j/2)] = array[i][j]
            
    return new_array

def compressAllCSV(full_csv):
    full_csv_compressed = np.empty([len(full_csv)], dtype=list)
    for i in range(len(full_csv)):
        full_csv_compressed[i] = compress(full_csv[i])
    
    return full_csv_compressed

def convertToImage(full_csv_compressed, files):
    for i in range(len(full_csv_compressed)):
        plt.imsave(f'{files[i][:len(files[i])-4]}_compressed.jpg', full_csv_compressed[i], cmap='gray')
    

files_sick = getFiles(url_sick)
files_healthy = getFiles(url_healthy)

full_csv_sick = fillWithCSVSick(files_sick)
full_csv_healthy = fillWithCSVHealthy(files_healthy)

full_csv_sick_compressed = compressAllCSV(full_csv_sick)
full_csv_healthy_compressed = compressAllCSV(full_csv_healthy)

#These 2 functions save the image files into the folder where the code is
#convertToImage(full_csv_sick_compressed, files_sick)
#convertToImage(full_csv_healthy_compressed, files_healthy)