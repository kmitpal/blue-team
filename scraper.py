import requests
from bs4 import BeautifulSoup
# url="https://realpython.github.io/fake-jobs/"
# html=requests.get(url)
# # print(html.text)

# s=BeautifulSoup(html.content,'html.parser')
# results=s.find(id='ResultsContainer')
# news=results.find_all('h2', class_='title is-5')
# print(news[0].text)

# for item in news:
#     print(item.text)




request = requests.get('https://www.bbc.com/news')
html=request.content

#create soup
soup=BeautifulSoup(html,'html.parser')
# print(soup.prettify())

def news_scrapper(keyword):
    news_list=[]

    #find all the headers in bbc home
    for h in soup.findAll('h3', class_='gs-c-promo-heading__title'):
        news_title = h.contents[0].lower()

        if news_title not in news_list:     
            if 'bbc' not in news_title:
                news_list.append(news_title)

    # print(news_list)  
    no_of_news = 0
    keyword_list=[]
    # goes thorugh the list and searches for the keyword
    for i, title in enumerate(news_list):
        text=''
        if keyword.lower() in title:
            text= '-----------------keyword'
            no_of_news+=1
            keyword_list.append(title)

        print(i +1,':',title,text)

    # prints the titles of the articles that contain the keyword
    print(f'\n-----------Total mentions of "{keyword}"={no_of_news}--------')
    for i,title in enumerate(keyword_list):
        print(i+1,':',title) 



news_scrapper('india')            











