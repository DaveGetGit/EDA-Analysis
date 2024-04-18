# News Data Analysis

This project is focused on Exploratory Data Analysis (EDA) of news data from Kaggle. It includes various Python scripts for data loading, data processing, and data visualization.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need to have Python installed on your machine. You can download Python from [here](https://www.python.org/downloads/).

### Installing

1. Clone the repository
   ```
   git clone <repository_url>
   ```
2. Navigate to the project directory
   ```
   cd <project_directory>
   ```
3. Create a virtual environment
   ```
   python -m venv .venv
   ```
4. Activate the virtual environment
   - On Windows, run:
     ```
     .venv\Scripts\activate
     ```
   - On Unix or MacOS, run:
     ```
     source .venv/bin/activate
     ```
5. Install the required packages
   ```
   pip install -r requirements.txt
   ```

## Running the Scripts

The project includes several Python scripts that you can run in a Jupyter notebook:

- `find_top_websites(news_data)`: Finds and plots the top news websites with the largest count of news articles.
- `find_high_traffic_websites(traffic_data)`: Finds and plots the top websites with the highest traffic.
- `plot_countries_with_most_media(file_path)`: Loads the domain location data, finds the countries with the most media organizations, and plots the data.
- `find_popular_articles(popular_countries_data)`: Finds and plots the top 10 countries mentioned in articles.

Please note that the `find_popular_articles` function uses Natural Language Toolkit (NLTK) library's named entity recognition (NER) which sometimes incorrectly classifies continents as 'Geopolitical Entities' (GPE). This might lead to continents being considered as countries in the final results.

## Data

The data used in this project is stored in large CSV files in the `data` directory. These files are not included in the repository due to their size.

## Built With

- [Python](https://www.python.org/)
- [Jupyter Notebook](https://jupyter.org/)
- [Matplotlib](https://matplotlib.org/)
- [NLTK](https://www.nltk.org/)
