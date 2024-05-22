# Facebook Marketplace Recommendation Ranking System

## 1. Data Cleaning

The data to be used in the models need to be cleaned in order to avoid compromising the prospective outputs. The dataset in the 'Products.csv' file, containing information about the listings, such as price, location and description had a high volume of null values that was removed, and one of it's columns; 'Price', was formatted to 'float64'. The pound sign as well as commas had to be removed to allow for the formatting, this is useful to enable arithmetic operations. The tabular dataset in the 'Products.csv' and 'image.csv' file had to be joined together in order to allow for the images to be appropriately labeled and in the process another column; 'category' was edited, the root category was extracted and the other details of each image in the category column was removed, this helped to simplify that paticular column and hence making the labelling straighforward. The next major cleaning that was carried out was on the images dataset in the images_fb file. The images had inconsistent pixel sizes and that had to be addressed and it was with the resize function as well as other attributes within useful modules such as PIL. The images was also formatted as 'RGB' to ensure standerdised channels.  

![Screenshot 2024-05-22 203959](https://github.com/martinsl01/facebook-marketplaces-recommendation-ranking-system41/assets/161818321/9d752eb1-d491-437b-9264-6bc729f0076c)

![Screenshot 2024-05-22 203929](https://github.com/martinsl01/facebook-marketplaces-recommendation-ranking-system41/assets/161818321/94d18a49-70b7-4c1c-b15f-d198cf3e027a)

![Screenshot 2024-05-22 203802](https://github.com/martinsl01/facebook-marketplaces-recommendation-ranking-system41/assets/161818321/08f6c1fe-17b3-447c-9732-30148877f768)
