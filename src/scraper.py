from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.webdriver.firefox.options import Options
import pandas as pd
import sys
import numpy as np
from pathlib import Path
# import time

TIMEOUT = 10
IMPLICIT_TIMEOUT = 10
LOAD_TIMEOUT = 30
ATTEMPTS = 0
options = Options()
options.add_argument("--headless")#setting to use a headless browser
driver = webdriver.Firefox(firefox_options = options) #instantiate Selenium
def new_browser(url):
    global driver
    global TIMEOUT
    global IMPLICIT_TIMEOUT
    global LOAD_TIMEOUT
    driver.implicitly_wait(IMPLICIT_TIMEOUT) #selenium max wait
    driver.set_page_load_timeout(LOAD_TIMEOUT)
    try:
        driver.get(url)
    except TimeoutException:
        LOAD_TIMEOUT += 10
        driver = new_browser(url)
    return driver

def get_beers(brewery_link):
    global TIMEOUT
    global IMPLICIT_TIMEOUT
    global driver
    global ATTEMPTS
    print("GETTING BEERS FROM ", brewery_link)
    driver = new_browser(brewery_link)
    try:#waits until page is loaded
        element = WebDriverWait(driver, TIMEOUT).until(
            EC.presence_of_element_located((By.XPATH, '//table[@id="brewer-beer-table"]/tbody/tr/td[not(em/label[@title="Currently out of production"])][not(em)]/strong/a'))
        )
    except TimeoutException:
        TIMEOUT += 2
        if ATTEMPTS == 3:
            ATTEMPTS = 0
            return None
        else:
            ATTEMPTS += 1
            return get_beers(brewery_link)
    finally:#gets beers urls
        beers = driver.find_elements_by_xpath('//table[@id="brewer-beer-table"]/tbody/tr/td[not(em/label[@title="Currently out of production"])][not(em)]/strong/a') #extract  beer links
        beer_links = []
        for beer in beers:
            beer_links.append(beer.get_attribute("href"))
        print('GOT THIS MANY BEERS', len(beer_links))
        # driver.quit() #closes unnused browser
        for beer_link in beer_links:
            get_beer(beer_link)

def get_beer(beer_link):
    global TIMEOUT
    global IMPLICIT_TIMEOUT
    global driver
    global ATTEMPTS
    global name, brewer, beer_style, score, rating_num, abv, ibu, est_cal, overall, style, about, photo_url, beer_url
    print('ACESSING', beer_link)
    try:#makes selenium waits until 60 seconds for element showing
        driver = new_browser(beer_link)
        element = WebDriverWait(driver, TIMEOUT).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="beerName"]'))
        )
        try:
            driver.find_element_by_xpath('//*[@id="beer-card-read-more"]').click()
        except ElementClickInterceptedException:
            TIMEOUT+=2
            if ATTEMPTS == 3:
                ATTEMPTS = 0
                return None
            else:
                ATTEMPTS += 1
                return get_beer(beer_link)
        element2 = WebDriverWait(driver, TIMEOUT).until(
            EC.presence_of_element_located((By.XPATH, '/html/body/div[1]/div/div[2]/div/div[2]/div/div[1]/div[1]/div/div[3]/div[1]/div/div[2]'))
        )
        print('getting data')
        try:
            name.append(driver.find_element_by_xpath('//*[@id="beerName"]').text)
        except NoSuchElementException:
            name.append(None)
        try:
            brewer.append(driver.find_element_by_xpath('//*[@id="brewerLink"]').text)
        except NoSuchElementException:
            brewer.append(None)
        try:
            beer_style.append(driver.find_element_by_xpath('//*[@id="styleLink"]').text)
        except NoSuchElementException:
            beer_style.append(None)
        try:
            score.append(driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div/div[2]/div/div[1]/div[1]/div/div[2]/div[1]/div[2]/div[2]/div/div[1]/span[1]').text)
        except NoSuchElementException:
            score.append(None)
        try:
            rating_num.append(driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div/div[2]/div/div[1]/div[1]/div/div[2]/div[1]/div[2]/div[2]/div/div[1]/span[3]/span[1]').text)
        except NoSuchElementException:
            rating_num.append(None)
        #alcohol p volumn
        try:
            abv.append(driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div/div[2]/div/div[1]/div[1]/div/div[2]/div[1]/div[2]/div[2]/div/div[2]/span[1]').text)
        except NoSuchElementException:
            abv.append(None)
        try:
            ibu.append(driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div/div[2]/div/div[1]/div[1]/div/div[2]/div[1]/div[2]/div[2]/div/div[3]/span[1]').text)
        except NoSuchElementException:
            ibu.append(None)
        try:
            est_cal.append(driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div/div[2]/div/div[1]/div[1]/div/div[2]/div[1]/div[2]/div[2]/div/div[4]/span[1]').text)
        except NoSuchElementException:
            est_cal.append(None)
        try:
            overall.append(driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div/div[2]/div/div[1]/div[1]/div/div[2]/div[1]/div[2]/div[1]/div[1]/span[2]').text)
        except NoSuchElementException:
            overall.append(None)
        try:
            style.append(driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div/div[2]/div/div[1]/div[1]/div/div[2]/div[1]/div[2]/div[1]/div[2]/span[1]').text)
        except NoSuchElementException:
            style.append(None)
        try:
            about.append(driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div/div[2]/div/div[1]/div[1]/div/div[3]/div[1]/div/div[2]').text)
        except NoSuchElementException:
            about.append(None)
        try:
            photo_url.append(driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div/div[2]/div/div[1]/div[1]/div/div[2]/div[2]/img').get_attribute("src"))
        except NoSuchElementException:
            photo_url.append(None)
        get_reviews()
        beer_url.append(beer_link)
    except TimeoutException:
        print('TIMED OUT, increasing TIMEOUT')
        IMPLICIT_TIMEOUT += 2
        get_beer(beer_link) #retry
    # finally:
            # driver.quit()
            # print('quitting driver')
def get_reviews():
    global TIMEOUT
    global IMPLICIT_TIMEOUT
    global driver
    global ATTEMPTS
    global aroma_avg, apparence_avg, taste_avg, palate_avg, overall_reviews_avg
    global aroma_med, apparence_med, taste_med, palate_med, overall_reviews_med
    aromas = []
    apparences = []
    tastes = []
    palates = []
    overalls = []
    try:
        elements = driver.find_elements_by_xpath('//*[@xmlns="http://www.w3.org/2000/svg"][@width="6"]')
        print ('elements', len(elements))
        for i in range(1, len(elements)+1):
            try:
                element = WebDriverWait(driver, TIMEOUT).until(
                    EC.element_to_be_clickable((By.XPATH, '/html/body/div[1]/div/div[2]/div/div[2]/div/div[1]/div[2]/div/div[2]/div['+str(i)+']/div/div[2]/div[4]/div/div'))
                )
                link = driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div/div[2]/div/div[1]/div[2]/div/div[2]/div['+str(i)+']/div/div[2]/div[4]/div/div')
                driver.execute_script('arguments[0].click();', link)
                element = WebDriverWait(driver, TIMEOUT).until(
                    EC.presence_of_element_located((By.XPATH, '/html/body/div[1]/div/div[2]/div/div[2]/div/div[1]/div[2]/div/div[2]/div['+str(i)+']/div/div[2]/div[4]/span/div/div[1]/div[2]'))
                )
                try:
                    aroma_r =  driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div/div[2]/div/div[1]/div[2]/div/div[2]/div['+str(i)+']/div/div[2]/div[4]/span/div/div[1]/div[2]').text
                    aroma_r = aroma_r.replace('/10', '')
                    aromas.append(int(aroma_r))
                    # print('aroma_r', int(aroma_r))
                except NoSuchElementException:
                    print ('no such element aroma_r')
                except TimeoutException:
                    print ('aroma timeout')
                try:
                    apparence_r =  driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div/div[2]/div/div[1]/div[2]/div/div[2]/div['+str(i)+']/div/div[2]/div[4]/span/div/div[2]/div[2]').text
                    apparence_r = apparence_r.replace('/5', '')
                    apparences.append(int(apparence_r))
                    # print('apparence', int(apparence_r))
                except NoSuchElementException:
                    print ('no such element apparence_r')
                except TimeoutException:
                    print ('apparence timeout')
                try:
                    taste_r =  driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div/div[2]/div/div[1]/div[2]/div/div[2]/div['+str(i)+']/div/div[2]/div[4]/span/div/div[3]/div[2]').text
                    taste_r = taste_r.replace('/10', '')
                    tastes.append(int(taste_r))
                    # print('taste_r', int(taste_r))
                except NoSuchElementException:
                    print ('no such element taste_r')
                except TimeoutException:
                    print ('taste timeout')
                try:
                    palate_r =  driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div/div[2]/div/div[1]/div[2]/div/div[2]/div['+str(i)+']/div/div[2]/div[4]/span/div/div[4]/div[2]').text
                    palate_r = palate_r.replace('/5', '')
                    palates.append(int(palate_r))
                    # print('palate', int(palate_r))
                except NoSuchElementException:
                    print ('no such element palate_r')
                except TimeoutException:
                    print ('palate timeout')
                try:
                    overall_r =  driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div/div[2]/div/div[1]/div[2]/div/div[2]/div['+str(i)+']/div/div[2]/div[4]/span/div/div[5]/div[2]').text
                    overall_r = overall_r.replace('/20', '')
                    overalls.append(int(overall_r))
                    # print('overall_r', int(overall_r))
                except NoSuchElementException:
                    print ('no such element overall_r')
                except TimeoutException:
                    print ('overall timeout')
            except TimeoutException:
                print('bug')

        #get averages
        aroma_avg.append(np.average(aromas))
        apparence_avg.append(np.average(apparences))
        taste_avg.append(np.average(tastes))
        palate_avg.append(np.average(palates))
        overall_reviews_avg.append(np.average(overalls))

        #get medians
        aroma_med.append(np.median(aromas))
        apparence_med.append(np.median(apparences))
        taste_med.append(np.median(tastes))
        palate_med.append(np.median(palates))
        overall_reviews_med.append(np.median(overalls))
    except NoSuchElementException:
        print ('no such element')
        TIMEOUT+=1
        if ATTEMPTS == 3:
            ATTEMPTS = 0
            return None
        else:
            ATTEMPTS += 1
            return get_reviews()

def insert_into_csv():
    global name, brewer, beer_style, score, rating_num, abv, ibu, est_cal, overall, style, about, photo_url, beer_url, aroma_avg, apparence_avg, taste_avg, palate_avg, overall_reviews_avg
    global aroma_med, apparence_med, taste_med, palate_med, overall_reviews_med
    print('creating CSV')
    item = {
            'name' : name,
            'brewer' : brewer,
            'beer_style' : beer_style,
            #MAXSCORE IS 5
            'score' : score,
            'rating_num' : rating_num,
            #alcohol p volumn
            'abv' : abv,
            'ibu' : ibu,
            'est_cal' : est_cal,
            'overall' : overall,
            'style' : style,
            'about' : about,
            'beer_url' : beer_url,
            'photo_url': photo_url,
            'aroma_avg': aroma_avg,
            'apparence_avg':apparence_avg,
            'taste_avg' : taste_avg,
            'palate_avg': palate_avg,
            'overall_reviews' : overall_reviews_avg,
            'aroma_med' : aroma_med,
            'apparence_med' : apparence_med,
            'taste_med' : taste_med,
            'palate_med' : palate_med,
            'overall_reviews_med' : overall_reviews_med,
            }
    df = pd.DataFrame.from_dict(item)
    # my_file = Path("/home/vplentz/Documentos/psr/beer/beer/scrapedBeers.csv") #set your
    if my_file.is_file():
        df.to_csv('data/scrapedBeers.csv', mode='a', header=False)
    else:
        df.to_csv('data/scrapedBeers.csv')
    #clear data
    name = []
    brewer = []
    beer_style = []
    score = []
    rating_num = []
    abv = []
    ibu = []
    est_cal = []
    overall = []
    style = []
    about = []
    photo_url = []
    beer_url = []
    #data from reviews
    #average
    aroma_avg = []
    apparence_avg = []
    taste_avg = []
    palate_avg = []
    overall_reviews_avg = []
    #medians
    aroma_med = []
    apparence_med = []
    taste_med = []
    palate_med = []
    overall_reviews_med = []

name = []
brewer = []
beer_style = []
score = []
rating_num = []
abv = []
ibu = []
est_cal = []
overall = []
style = []
about = []
photo_url = []
beer_url = []
#data from reviews
#average
aroma_avg = []
apparence_avg = []
taste_avg = []
palate_avg = []
overall_reviews_avg = []
#medians
aroma_med = []
apparence_med = []
taste_med = []
palate_med = []
overall_reviews_med = []

print(sys.argv)
brewery_start_url = None
if len(sys.argv) > 1:
    brewery_start_url = sys.argv[1]
print('DONT WORRY, ITS NOT FREEZED')
driver = new_browser('https://www.ratebeer.com/breweries/brazil/0/31/')

try:#makes selenium waits until 60 seconds for element showing
    element = WebDriverWait(driver, TIMEOUT).until(
        EC.presence_of_element_located((By.XPATH, '/html/body/div[1]/div[3]/div[1]/div[2]/div/div[1]/table/tbody/tr/td/a'))
    )
except TimeoutException:
    print('Load breaweries timeout')
    exit()
finally:
    print('GETTING BREWERIES')
    try:
        breweries = driver.find_elements_by_xpath('/html/body/div[1]/div[3]/div[1]/div[2]/div/div[1]/table/tbody/tr/td/a') #extract breweries
    except NoSuchElementException:
        print('Couldnt find the breweries table, exit')
        exit()
    brewerie_links = []
    for brewery in breweries[::2]: #gets links
        brewerie_links.append(brewery.get_attribute("href"))
    if brewery_start_url != None: #check if has a argv brewery link
        brewerie_links = brewerie_links[brewerie_links.index(brewery_start_url):]
    print('GOT ', len(brewerie_links), 'BREWERIES')
    # print(breweries)
    # driver.quit() #closes unnused browser
    for brewery_link in brewerie_links: #access links
        get_beers(brewery_link)
        insert_into_csv()
