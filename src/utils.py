import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from collections import Counter
from multiprocessing import Pool
from tqdm import tqdm


def find_top_websites(news_data, url_column='url', top_count=10):
    """
    Get the top [top_count] websites with the highest article counts.
    """
    news_data['domain'] = news_data[url_column].apply(
        lambda x: x.split('/')[2])

    # Count occurrences of each domain
    domain_counts = news_data['domain'].value_counts()

    top_domains = domain_counts.head(top_count)
    return top_domains


def find_high_traffic_websites(news_data, top_count=10):
    """
    Get websites with high reference IPs (assuming the IPs are the number of traffic).
    """
    traffic_per_domain = news_data.groupby(['Domain'])['RefIPs'].sum()
    traffic_per_domain = traffic_per_domain.sort_values(ascending=False)
    return traffic_per_domain.head(top_count)


def find_countries_with_most_media(news_data, top_count=10):
    """
    Get the top countries with the most media outlets.
    """
    media_per_country = news_data['Country'].value_counts()
    media_per_country = media_per_country.sort_values(ascending=False)
    return media_per_country.head(top_count)


# first download required nltk packages

def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')


count = 0


def extract_countries_from_article_content(article):
    index, row = article
    text = row['content']
    # tokenize every text into words
    words = word_tokenize(text)
    # part of speech tagging; this means tag every word; with noun,verb ...
    tagged_words = pos_tag(words)
    # get named entities from the tagged lists
    named_entities = ne_chunk(tagged_words)
    # GPE stands for 'Geopolitical Entity' in our case country name
    countries = [chunk[0] for chunk in list(named_entities) if hasattr(
        chunk, 'label') and chunk.label() == 'GPE']

    return countries


def find_popular_articles(popular_countries_data):
    print('downloading nltk resources ...')
    download_nltk_resources()
    print('finished downloading resources ...')
    print('loading data ...')
    df = popular_countries_data
    print('starting processing this might take a while ...')
    # since we have a lot of data we need to parallize the process

    # Maximum number of rows to process
    max_rows = 100000  # len(df)
    print(f'max rows is: {max_rows}')
    processed_count = 0
    # Apply function to each article in parallel with tqdm for progress bar
    with Pool() as pool:
        results = []
        for countries in tqdm(pool.imap(extract_countries_from_article_content, df.iterrows()), total=len(df)):
            # Append the results
            results.append(countries)

            # Increment processed_count
            processed_count += 1

            # Check if maximum number of rows processed
            if processed_count >= max_rows:
                print("Maximum number of rows processed. Stopping pool.")
                break
    print('done processing!')
    # Flatten the list of results
    all_countries = [country for country in all_countries if country.lower(
    ) not in [continent.lower() for continent in continents]]

    # List of continents to filter out
    continents = ['African', 'Antarctica', 'Asia', 'Europe',
                  'North America', 'Australia', 'South America']

    # Filter out continents from all_countries
    all_countries = [
        country for country in all_countries if country not in continents]

    # Count occurrences of each country
    print("debug printing count...")
    country_counts = Counter(all_countries)
    print(country_counts.most_common(3))
    return country_counts.most_common(10)


def website_sentiment_distribution(data):
    """
    Calculate the sentiment distribution of websites based on the given data.

    Args:
        data (DataFrame): The input data containing the website information.

    Returns:
        DataFrame: The sentiment counts for each website along with mean and median values.
    """
    sentiment_counts = data.groupby(
        ['source_name', 'title_sentiment']).size().unstack(fill_value=0)
    sentiment_counts['Total'] = sentiment_counts.sum(axis=1)

    # Calculate mean and median sentiment counts for each domain
    sentiment_counts['Mean'] = sentiment_counts[[
        'Positive', 'Neutral', 'Negative']].mean(axis=1)
    sentiment_counts['Median'] = sentiment_counts[[
        'Positive', 'Neutral', 'Negative']].median(axis=1)

    # Display the sentiment counts along with mean and median
    print("Sentiment counts with mean and median:")
    print(sentiment_counts)
    return sentiment_counts


def webiste_sentiment(data):
    sentiment_counts = data.groupby(
        ['source_name', 'title_sentiment']).size().unstack(fill_value=0)
    return sentiment_counts
