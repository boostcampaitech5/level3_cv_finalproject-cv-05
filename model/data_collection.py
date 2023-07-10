"""
search_term: 개체 클래스 명
공백을 제거하고 사이에 +를 추가     Ex) search_term Samsung+TV
number_of_images: 해당 개체 클래스에서 원하는 이미지 수
starting_number: 인덱싱은 이전에 다운로드한 이미지를 덮어쓰지 않아야 함
이전에 스크랩한 데이터에 추가할 starting_number를 설정 가능
"""

import time
import os
from selenium import webdriver
import bs4
import requests
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

# creating a directory to save images
folder_name = "../images"  # "../2_Data_Annotation/images"
if not os.path.isdir(folder_name):
    os.makedirs(folder_name)


def download_image(url, folder_name, num):
    # write image to file
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(folder_name, str(num) + ".jpg"), "wb") as file:
            file.write(response.content)


chrome_options = Options()
chrome_options.add_argument("--window-size=1920,1200")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)


def scrape_images(search_term, number_of_images, starting_number):
    SEARCH_URL = f"https://www.google.com/search?q={search_term}&source=lnms&tbm=isch"
    driver.get(SEARCH_URL)

    # a = input("Waiting...")

    # Scrolling all the way up
    driver.execute_script("window.scrollTo(0, 0);")  # 제일 위로 올림, 스크롤 내려가서 검색결과 안나오게

    page_html = driver.page_source
    pageSoup = bs4.BeautifulSoup(page_html, "html.parser")
    containers = pageSoup.findAll("div", {"class": "isv-r PNCib MSM1fd BUooTd"})

    len_containers = len(containers)
    print(f"Found {len_containers} containers")

    # driver.find_element(By.XPATH, """//*[@id="islrg"]/div[1]/div[1]""").click()  # 첫번째 검색결과 이미지 클릭
    count = starting_number
    end = count + number_of_images + 1
    for i in range(1, len_containers):
        if i % 25 == 0 or count == end:  # i % 25 == 0 >>> 25번째는 왜?
            continue
        else:
            di = i // 50
            ni = i % 50
            # 0,50 / 51 - 0,50 / 52, .. - 0,100
            if di == 0:
                xPath = f"""//*[@id="islrg"]/div[1]/div[{i}]"""
                # //*[@id="islrg"]/div[1]/div[1]/a[1]/div[1]/img
                # //*[@id="islrg"]/div[1]/div[51]/div[50]
                # URL of the small preview image

                previewImageXPath = f"""//*[@id="islrg"]/div[1]/div[{i}]/a[1]/div[1]/img"""  # 클릭전 작은 이미지  src확보 ?왜필요
                previewImageElement = driver.find_element(By.XPATH, previewImageXPath)
                previewImageURL = previewImageElement.get_attribute("src")

                driver.find_element(
                    By.XPATH, xPath
                ).click()  # 작은 이미지 클릭   //////////////여기서 오류 발생 가장 처음 이미지 클릭이 안되는데 이유 모름  > 위에서 쓸데없이 한번 클릭해서 그런거였음
            else:
                xPath = f"""//*[@id="islrg"]/div[1]/div[{50+di}]/div[{ni}]"""

                previewImageXPath = (
                    f"""//*[@id="islrg"]/div[1]/div[{50+di}]/div[{ni}]/a[1]/div[1]/img"""  # 클릭전 작은 이미지  src확보 ?왜필요
                )
                previewImageElement = driver.find_element(By.XPATH, previewImageXPath)
                previewImageURL = previewImageElement.get_attribute("src")

                driver.find_element(By.XPATH, xPath).click()

            timeStarted = time.time()
            while True:
                # imageElement = wait.until(EC.visibility_of_element_located((By.XPATH,'//*[@id="Sva75c"]/div[2]/div[2]/div[2]/div[2]/c-wiz/div/div/div/div[3]/div[1]/a/img[1]',)))
                imageElement = driver.find_element(
                    By.XPATH,
                    """//*[@id="Sva75c"]/div[2]/div[2]/div[2]/div[2]/c-wiz/div/div/div/div[3]/div[1]/a/img[1]""",
                )  # 클릭해서 옆에 뜬 이미지 링크
                imageURL = imageElement.get_attribute("src")

                if imageURL != previewImageURL:
                    break

                else:
                    # timeout functionality
                    current_time = time.time()

                    if current_time - timeStarted > 10:
                        print("Timeout will move on to next image")
                        break
            # Downloading image
            try:
                download_image(imageURL, folder_name, count)
                count += 1
                print("Downloaded element %s out of %s total. URL: %s" % (i, len_containers + 1, imageURL))
            except:
                print("Couldn't download an image %s, continuing downloading the next one" % (i))


# scrape_images("검색어","찾을 개수","시작 인덱스")
scrape_images("magicmouse+A1657", 100, 0)

# scrape_images("fan", 100, 100)
# scrape_images("microwave",100,200)

time.sleep(10)
driver.quit()
