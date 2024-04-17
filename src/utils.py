def find_top_websites(news_data, url_column='url', top_count=10):
    """
    Get the top [top_count] websites with the highest article counts.
    """
    news_data['domain'] = news_data[url_column].apply(lambda x: x.split('/')[2])

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
