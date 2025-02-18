import pandas as pd
from bs4 import BeautifulSoup as Soup
import requests

site_url = 'https://habr.com/ru/companies/vtb/profile/'
links = [site_url]

constructor = {
    'Название': [],
    'Рейтинг': [],
    'Описание': [],
    'Сфера деятельности': [],
    'Дата публрикации': []
}

for link in links:
    response = requests.get(link)
    soup = Soup(response.content, 'html.parser')
    items = soup.find_all('div', class_='pull-down')
    for item in items:
        title = item.find('a', class_='tm-company-card__name').text.strip()
        reiting = item.find('span', class_='tm-votes-lever__score-counter').text.strip()
        info = item.find('span', class_='tm-company-profile__content').text.strip()
        sfera = item.find('div', class_='tm-company-profile__categories').text.strip()
        datepub = item.find('dd', class_='tm-description-list__body').text.strip()
        constructor['Название'] += [title]
        constructor['Рейтинг'] += [reiting]
        constructor['Описание'] += [info]
        constructor['Сфера деятельности'] += [sfera]
        constructor['Дата публрикации'] += [datepub]

pd.DataFrame(constructor).to_csv('./DataFrame.csv', encoding="utf-8", index=False)

osnovadf = pd.read_csv('DataFrame.csv')





url_list = ['https://habr.com/ru/companies/avito/profile/', 'https://habr.com/ru/companies/vtb/profile/',
            'https://habr.com/ru/companies/vk/profile/', 'https://habr.com/ru/companies/vk/profile/']
i = 0
while i != len(url_list):
    site_url = url_list[i]
    links = [site_url]

    constructor = {
        'Название': [],
        'Рейтинг': [],
        'Описание': [],
        'Сфера деятельности': [],
        'Дата публикации': []
    }
    for link in links:
        response = requests.get(link)
        soup = Soup(response.content, 'html.parser')
        items = soup.find_all('div', class_='pull-down')
        for item in items:
            title = item.find('a', class_='tm-company-card__name').text.strip()
            reiting = item.find('span', class_='tm-votes-lever__score-counter').text.strip()
            info = item.find('span', class_='tm-company-profile__content').text.strip()
            sfera = item.find('div', class_='tm-company-profile__categories').text.strip()
            datepub = item.find('dd',
                                class_='tm-description-list__body tm-description-list__body tm-description-list__body_variant-columns-info').text.strip()
            constructor['Название'] += [title]
            constructor['Рейтинг'] += [reiting]
            constructor['Описание'] += [info]
            constructor['Сфера деятельности'] += [sfera]
            constructor['Дата публикации'] += [datepub]
        pd.DataFrame(constructor).to_csv('./DataFrame2.csv', encoding="utf-8", index=False)
        i += 1

povtordf = pd.read_csv('DataFrame2.csv')

## просмотр датафрейма
povtordf

