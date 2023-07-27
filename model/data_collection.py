import time
import os
from selenium import webdriver
import bs4
import yaml
import requests
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from argparse import ArgumentParser
from selenium.webdriver.chrome.service import Service as ChromiumService
from webdriver_manager.core.utils import ChromeType

def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, yaml_path):
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--class_name",type=str, default="text")
    parser.add_argument("--model_name",type=str, default="text")
    args = parser.parse_args()
    return args
def download_image(url, folder_name, num):
    # write image to file
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(folder_name, str(num) + ".jpg"), "wb") as file:
            file.write(response.content)


def scrape_images(driver, folder_name, search_term, number_of_images, starting_number):
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
    end = count + number_of_images
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


def main(class_name, model_name):
    # creating a directory to save images
    paths = "../model/new_dataset"
    if not os.path.isdir(paths):
        os.makedirs(paths)
    folder_name = paths + "/user_images"
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1200")    
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    
    driver = webdriver.Chrome(service=ChromiumService(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()), options=chrome_options)#ChromeDriverManager().install()

    # scrape_images("검색어","찾을 개수","시작 인덱스")
    scrape_images(driver, folder_name, class_name + "+" + model_name, 5, 0)

    time.sleep(10)
    driver.quit()


if __name__ == "__main__":
    args = parse_args()
    class_name = args.class_name #"mouse"
    model_name = args.model_name #"A1657"

    main(class_name, model_name)

    yaml_path = "../model/config.yaml"  # config 수정
    config = load_config(yaml_path)
    config["class_name"] = class_name
    save_config(config, yaml_path)
