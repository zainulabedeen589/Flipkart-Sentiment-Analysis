import requests
from bs4 import BeautifulSoup
import re
import csv
import os

Reviewer_Name = []
Reviewer_Rating = []
Review_Title = []
Review_Text = []
Place_of_Review = []
Date_of_Review = []
Up_Votes = []
Down_Votes = []

for i in range(2, 1000):
    url = 'https://www.flipkart.com/yonex-mavis-350-nylon-shuttle-yellow/product-reviews/itmfcjdyhnghfyey?pid=STLEFJ7UFQGRUUR3&lid=LSTSTLEFJ7UFQGRUUR3SUDA2S&aid=overall&certifiedBuyer=false&sortOrder=MOST_HELPFUL&page=' + str(i)
    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'lxml')
        name = soup.find_all('p', class_='_2sc7ZR _2V5EHH')
        for i in name:
            n = i.text.strip()
            Reviewer_Name.append(n)
        
        # Find elements with class representing ratings 3, 4, and 5
        rating_class1 = soup.find_all('div', class_='_3LWZlK _1BLPMq')

        # Find elements with class representing ratings 1 and 2
        rating_class2 = soup.find_all('div', class_='_3LWZlK _32lA32 _1BLPMq')
        
        rating_class3 = soup.find_all('div', class_='_3LWZlK _1rdVr6 _1BLPMq')

        # Iterate over elements found for class 1
        for elem in rating_class1:
            r = elem.text.strip()
            # Add rating to Reviewer_Rating list
            Reviewer_Rating.append(r)

        # Iterate over elements found for class 2
        for elem in rating_class2:
            r = elem.text.strip()
            # Add rating to Reviewer_Rating list
            Reviewer_Rating.append(r)
        
        for elem in rating_class3:
            r = elem.text.strip()
            # Add rating to Reviewer_Rating list
            Reviewer_Rating.append(r)


        title = soup.find_all('p', class_='_2-N8zT')
        for i in title:
            t = i.text.replace('READ MORE', '').strip() if 'READ MORE' in i.text else i.text.strip()
            Review_Title.append(t)

        text = soup.find_all('div', class_='t-ZTKy')
        for i in text:
            te = i.text.strip().replace('READ MORE', '').strip() if 'READ MORE' in i.text else i.text.strip()
            Review_Text.append(te)

        place = soup.find_all('p', class_='_2mcZGG')
        for i in place:
            p = i.text.strip()
            Place_of_Review.append(p)

        date = soup.find_all('p', class_='_2sc7ZR')
        for i in date:
            text = i.text.strip()
            # Check if the text contains a comma and has a year
            if ',' in text and any(char.isdigit() for char in text):
                Date_of_Review.append(text)

        up_elements = soup.find_all('div', class_='_1LmwT9')
        for element in up_elements:
            span_elements = element.find_all('span', class_='_3c3Px5')
            for span_element in span_elements:
                text = span_element.text.strip()
                # Use regular expression to extract only numeric values
                numeric_value = re.search(r'\d+', text)
                if numeric_value:
                    Up_Votes.append(numeric_value.group())
                    # Break the loop after finding the first numeric value
                    break
            else:
                Up_Votes.append(None)  # Append None if no numeric value found

        down = soup.find_all('div', class_='_1LmwT9 pkR4jH')
        for i in down:
            text = i.text.strip()
            # Use regular expression to extract only numeric values
            numeric_value = re.search(r'\d+', text)
            if numeric_value:
                Down_Votes.append(numeric_value.group())
            else:
                Down_Votes.append(None)  # Append None if no numeric value found
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while fetching page {i}: {e}")

# Truncate Up_Votes list to half its length
Up_Votes = Up_Votes[:len(Up_Votes)//2]

# Ensure all lists have the same length by padding with None where necessary
max_length = max(len(Reviewer_Name), len(Reviewer_Rating), len(Review_Title), len(Review_Text), len(Place_of_Review), len(Date_of_Review), len(Up_Votes), len(Down_Votes))
Reviewer_Name += [None] * (max_length - len(Reviewer_Name))
Reviewer_Rating += [None] * (max_length - len(Reviewer_Rating))
Review_Title += [None] * (max_length - len(Review_Title))
Review_Text += [None] * (max_length - len(Review_Text))
Place_of_Review += [None] * (max_length - len(Place_of_Review))
Date_of_Review += [None] * (max_length - len(Date_of_Review))
Up_Votes += [None] * (max_length - len(Up_Votes))
Down_Votes += [None] * (max_length - len(Down_Votes))

# Create a list of dictionaries

current_directory = os.getcwd()
file_path = os.path.join(current_directory, 'ProjectReviews_from_Flipkart.csv')

reviews = []
for i in range(max_length):
    review = {
        'Reviewer Name': Reviewer_Name[i] if i < len(Reviewer_Name) else None,
        'Reviewer Rating': Reviewer_Rating[i] if i < len(Reviewer_Rating) else None,
        'Review Title': Review_Title[i] if i < len(Review_Title) else None,
        'Review Text': Review_Text[i] if i < len(Review_Text) else None,
        'Place of Review': Place_of_Review[i] if i < len(Place_of_Review) else None,
        'Date of Review': Date_of_Review[i] if i < len(Date_of_Review) else None,
        'Up Votes': Up_Votes[i] if i < len(Up_Votes) else None,
        'Down Votes': Down_Votes[i] if i < len(Down_Votes) else None
    }
    reviews.append(review)

# Write to CSV
with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Reviewer Name', 'Reviewer Rating', 'Review Title', 'Review Text',
                  'Place of Review', 'Date of Review', 'Up Votes', 'Down Votes']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(reviews)
