	import time
	import selenium 
	from selenium import webdriver
	from selenium.webdriver.common.keys import Keys
	import requests
	import os 

	def scrape_and_save(search_name,web_driver = wd):
		DRIVER_PATH = '/home/trey/Scraping-Images/chromedriver'
		wd = webdriver.Chrome(executable_path=DRIVER_PATH)
		search_url = "https://www.google.com/search?q={q}&sxsrf=ALeKk02zAb9RaNNb-qSenTEJh1i2XX480w:1613489053802&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjChJqP2-7uAhVyTTABHdX0CPoQ_AUoAXoECAcQAw&biw=767&bih=841"
		wd.get(search_url.format(q='Dogs'))
		body = wd.find_element_by_tag_name("body")
    		body.send_keys(Keys.PAGE_DOWN)
    		time.sleep(2.5)
    		main = wd.find_elements_by_class_name("rg_i.Q4LuWd")
#wd.quit()
    		links = [main[i].get_attribute('src') for i in range(len(main))]
    		imagess = []
    		for image in links:
    # making sure type of data is an image containing the string google encodes for an image
        		if type(image) == str:
        # checking conditional text so Image library can open based on based64 conversion
        		if image[0:4] == 'data':
            # replace text so we can open it
                		new = image.replace("data:image/jpeg;base64,","")
            # adding equals at the end for decoding
                		if new[-2:] != '==':
                    			new_edit = new + '=='
                # image becomes Image object
                    			new_image = (Image.open(BytesIO(base64.b64decode(new_edit)))).resize((150,150))
                    			imagess.append(new_image)
                		else:
                    			new_image = (Image.open(BytesIO(base64.b64decode(new)))).resize((150,150))
                    			imagess.append(new_image)
		if image[0:4] == 'http':
                	new = requests.get(image)
                	new_image = Image.open(BytesIO(image.content))
                	imagess.append(new_image)
# creating directories, saving images, and creating names for images as jpeg files
	os.mkdir('Images')
	os.mkdir('Images/regular')
	index = 0 
	for i in imagess:
		i.save('Images/regular'+'/'+string(index)+'.jpeg')
		index += 1


#
