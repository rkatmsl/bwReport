import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from collections import Counter
import ast
import datetime
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
import random

# Sample data for Snippet Link Reach Shares
snippet_data = [
    {"title": "Aramco boosts renewable energy with new solar project", "url": "https://example.com/1", "reach": 120000, "shares": 4500},
    {"title": "Saudi Aramco partners with tech firms for AI innovation", "url": "https://example.com/2", "reach": 95000, "shares": 3200},
    {"title": "Aramco's new policy on carbon neutrality", "url": "https://example.com/3", "reach": 80000, "shares": 2800},
    {"title": "Energy transition: Aramco's $10B investment", "url": "https://example.com/4", "reach": 65000, "shares": 2100},
    {"title": "Aramco competes with Reliance in Asian markets", "url": "https://example.com/5", "reach": 70000, "shares": 1900},
]
snippet_df = pd.DataFrame(snippet_data)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_excel('bw.xlsx')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['matchPositions'] = df['matchPositions'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x != '' else [])
    df['professions'] = df['professions'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x != '' else [])
    df['tags'] = df['tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x != '' else [])
    df['insightsHashtag'] = df['insightsHashtag'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x != '' else [])
    df['classifications'] = df['classifications'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x != '' else [])
    df.fillna({
        'author': 'Unknown',
        'impressions': 0,
        'twitterAuthorId': 0,
        'reachEstimate': 0,
        'dailyVisitors': 0,
        'monthlyVisitors': 0,
        'facebookLikes': 0,
        'facebookComments': 0,
        'facebookShares': 0,
        'twitterLikeCount': 0,
        'twitterRetweets': 0,
        'twitterReplyCount': 0,
        'instagramLikeCount': 0,
        'instagramCommentCount': 0,
        'instagramInteractionsCount': 0,
        'linkedinLikes': 0,
        'linkedinComments': 0,
        'linkedinShares': 0,
        'tiktokLikes': 0,
        'tiktokComments': 0,
        'tiktokShares': 0,
        'twitterFollowers': 0,
        'instagramFollowerCount': 0,
        'linkedinImpressions': 0,
        'sentiment': 'neutral',
        'domain': 'Unknown',
        'contentSource': 'Unknown'
    }, inplace=True)
    df['engagementScore'] = (
        df['facebookLikes'] + df['facebookComments'] + df['facebookShares'] +
        df['twitterLikeCount'] + df['twitterRetweets'] + df['twitterReplyCount'] +
        df['instagramLikeCount'] + df['instagramCommentCount'] + df['instagramInteractionsCount'] +
        df['linkedinLikes'] + df['linkedinComments'] + df['linkedinShares'] +
        df['tiktokLikes'] + df['tiktokComments'] + df['tiktokShares']
    )
    return df

def get_top_words(texts, n=10):
    words = []
    for text in texts:
        if pd.notna(text) and isinstance(text, str):
            words.extend(re.findall(r'\b\w+\b', text.lower()))
    return Counter(words).most_common(n)

def generate_wordcloud(tags):
    if not tags:
        return None
    tag_counts = Counter(tags)
    wordcloud = WordCloud(width=800, height=400, background_color='white', min_font_size=10).generate_from_frequencies(tag_counts)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return img_str

# Load data
df = load_data()

# Set page configuration
st.set_page_config(
    page_title="Aramco India Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Apply filters
filtered_df = df.copy()

@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    random.seed(42)
    
    dates = pd.date_range(end=datetime.datetime.now(), periods=30, freq='D')
    
    data = []
    
    sample_authors = [
        "OilGasInsider", "EnergyAnalyst", "PetroleumWatch", "IndustryExpert", 
        "SaudiNewsHub", "MideastEnergy", "GlobalOilNews", "EnergyReporter",
        "CrudeOilDaily", "PetroUpdate", "EnergyMarkets", "OilPriceWatch"
    ]
    
    sample_domains = [
        "twitter.com", "linkedin.com", "reuters.com", "bloomberg.com", 
        "oilprice.com", "arabnews.com", "spa.gov.sa", "energyvoice.com",
        "offshore-technology.com", "rigzone.com", "facebook.com", "instagram.com"
    ]
    
    sample_sentiments = ["positive", "neutral", "negative"]
    sample_emotions = ["excitement", "trust", "anticipation", "neutral", "concern", "optimism"]
    
    sample_titles = [
        "Aramco announces major expansion in renewable energy projects",
        "Saudi Aramco reports strong Q4 earnings amid oil price recovery",
        "Aramco partners with tech giants for digital transformation",
        "New discovery increases Saudi Arabia's oil reserves",
        "Aramco invests $15B in blue hydrogen production",
        "Saudi Aramco signs MOU for carbon capture technology",
        "Aramco's IPO continues to attract global investors",
        "Energy transition: Aramco's strategy for sustainable future",
        "Aramco and Shell announce joint venture in petrochemicals",
        "Saudi Aramco expands presence in Asian markets"
    ]
    
    sample_hashtags = [
        "#Aramco", "#SaudiAramco", "#Oil", "#Energy", "#Sustainability", 
        "#CleanEnergy", "#Innovation", "#Technology", "#Investment", "#Growth",
        "#Renewable", "#Hydrogen", "#CarbonNeutral", "#DigitalTransformation"
    ]
    
    sample_topics = [
        "Oil Production", "Renewable Energy", "Digital Innovation", "Sustainability",
        "Market Expansion", "Technology Partnership", "Financial Performance",
        "Energy Transition", "Investment", "Climate Action"
    ]
    
    for i in range(300):  # Generate 300 sample posts
        date = random.choice(dates)
        author = random.choice(sample_authors)
        domain = random.choice(sample_domains)
        sentiment = random.choice(sample_sentiments)
        
        # Weighted sentiment distribution (more positive for Aramco)
        sentiment_weights = [0.5, 0.35, 0.15]  # positive, neutral, negative
        sentiment = np.random.choice(sample_sentiments, p=sentiment_weights)
        
        data.append({
            'date': date,
            'author': author,
            'domain': domain,
            'title': random.choice(sample_titles),
            'sentiment': sentiment,
            'emotion': random.choice(sample_emotions),
            'reach': random.randint(1000, 100000),
            'impressions': random.randint(5000, 500000),
            'engagement': random.randint(10, 1000),
            'hashtags': random.sample(sample_hashtags, random.randint(1, 3)),
            'topics': random.sample(sample_topics, random.randint(1, 2)),
            'followers': random.randint(1000, 50000),
            'url': f"https://example.com/post_{i}"
        })
    
    return pd.DataFrame(data)

df2 = generate_sample_data()
cutoff_date = datetime.datetime.now() - datetime.timedelta(days=30)
filtered_df2 = df2[df2['date'] >= cutoff_date]

st.markdown("""
    <style>
            .brand-color{
            color: #2688b5 !important;
            }
        .logo {
            width: 100px;  /* Adjust as needed */
        }

    .main-title {
        text-align: left;
        padding-bottom: 20px;
    }
    .date-header {
        text-align: end;
        font-size: 18px;
        font-weight: bold;
        padding: 10px 0;
    }
    .price-box {
        border-radius: 8px;
        padding: 15px;
        background-color: #2688b5;
        width: max-content;
        margin: 10px 0;
        margin-left: auto;
    }
    .price-text {
        font-size: 16px;
        font-weight: bold;
        color: #fff;
        margin: 5px 0;
    }
    .column-padding {
        padding: 0 20px;
        display: flex;
        justify-content: flex-end; /* Ensures the content in the column is right-aligned */
    }
    </style>
""", unsafe_allow_html=True)

logo_col, box_col = st.columns([1, 1])

with logo_col:
    st.markdown('<img class="logo" src="https://www.aramco.com/-/jssmedia/project/aramcocom/aramco-logo.png" alt="Logo">', unsafe_allow_html=True)
with box_col:
    st.markdown("""
        <div class="price-box">
            <div class="price-text">Crude Oil (WTI): $82.45</div>
            <div class="price-text">Brent Oil: $87.20</div>
        </div>
    """, unsafe_allow_html=True)



col1, col2 = st.columns([3, 2])

with col1:
    st.markdown('<h1 class="main-title brand-color">Daily Reports Aramco</h1>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="date-header">23 May 2025</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric('Total Mentions', f"{len(filtered_df):,}")
col2.metric('Unique Authors', f"{filtered_df['author'].nunique():,}")
col3.metric('Total Reach', f"{filtered_df['reachEstimate'].sum() / 1_000_000:.1f}M" if filtered_df['reachEstimate'].sum() >= 1_000_000 else f"{filtered_df['reachEstimate'].sum():,}")
col4.metric('Impressions', f"{filtered_df['impressions'].sum() / 1_000_000:.1f}M" if filtered_df['impressions'].sum() >= 1_000_000 else f"{filtered_df['impressions'].sum():,}")
col5.metric('X Reposts', f"{filtered_df['twitterRetweets'].sum():,}")
col6.metric('Engagement Score', f"{filtered_df['engagementScore'].sum():,}")

mention_volume = filtered_df.groupby(filtered_df['date'].dt.date).size().reset_index(name='mentions')
mention_volume['7-day MA'] = mention_volume['mentions'].rolling(window=7).mean()
fig_volume = go.Figure()
fig_volume.add_trace(go.Scatter(x=mention_volume['date'], y=mention_volume['mentions'], mode='lines+markers', name='Daily Mentions'))
fig_volume.add_trace(go.Scatter(x=mention_volume['date'], y=mention_volume['7-day MA'], mode='lines', name='7-day Moving Average', line=dict(dash='dash')))
fig_volume.update_layout(
    # title='Mention Volume Over Time',
    xaxis_title='Date',
    yaxis_title='Number of Mentions',
    hovermode='x unified',
    template='plotly_white'
)
st.plotly_chart(fig_volume, use_container_width=True)


col_news, col_pie = st.columns([1, 1])


snippet_data = [
    {"title": "Aramco boosts renewable energy with new solar project", "url": "https://example.com/1", "reach": 120000, "shares": 4500},
    {"title": "Saudi Aramco partners with tech firms for AI innovation", "url": "https://example.com/2", "reach": 95000, "shares": 3200},
    {"title": "Aramco's new policy on carbon neutrality", "url": "https://example.com/3", "reach": 80000, "shares": 2800},
    {"title": "Energy transition: Aramco's $10B investment", "url": "https://example.com/4", "reach": 65000, "shares": 2100},
    {"title": "Aramco competes with Reliance in Asian markets", "url": "https://example.com/5", "reach": 70000, "shares": 1900},
]
snippet_df = pd.DataFrame(snippet_data)


with col_news:
    st.markdown('<h3 class=" brand-color">Top Tweets</h3>', unsafe_allow_html=True)
    snippet_df['Reach (Formatted)'] = snippet_df['reach'].apply(lambda x: f"{x/1000:.0f}K" if x < 1000000 else f"{x/1000000:.1f}M")
    table_data = snippet_df[['title', 'Reach (Formatted)']].head(5)
    table_data.columns = ['Snippet', 'Reach']
    
    html = "<table style='width: 100%'><tr><th>Snippet</th><th>Reach</th></tr>"
    for index, row in table_data.iterrows():
        snippet = row['Snippet']
        reach = row['Reach']
        html += f"<tr><td>{snippet}</td><td>{reach}</td></tr>"
    html += "</table>"
    st.write(html, unsafe_allow_html=True)

with col_pie:
    domain_reach = filtered_df2.groupby('domain')['reach'].sum().sort_values(ascending=False).head(5)
    
    fig_domains = px.pie(
        values=domain_reach.values,
        names=domain_reach.index,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_domains.update_traces(textinfo='percent+label', pull=[0.1, 0, 0])
    st.plotly_chart(fig_domains, use_container_width=True)


st.markdown('<h3 class=" brand-color">Sentiment and Themes</h3>', unsafe_allow_html=True)
col_sentiment, col_emotions = st.columns(2)
     
industry_mentions = {
    'Policy and Regulatory': 35,
    'Projects and Infra': 25,
    'Energy Transition': 15,
    'Wind': 15,
    'Tech and Innovation': 5
}

industry_df = pd.DataFrame(list(industry_mentions.items()), columns=['Themes', 'Mentions'])

with col_sentiment:
    # st.subheader('Sentiment Distribution')
    sentiment_counts = filtered_df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig_sentiment = px.pie(
        sentiment_counts,
        values='Count',
        names='Sentiment',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_sentiment.update_traces(textinfo='percent+label', pull=[0.1, 0, 0])
    st.plotly_chart(fig_sentiment, use_container_width=True)

with col_emotions:
    # st.subheader('Themes Distribution')

    fig_competitors = px.pie(
        industry_df,
        values='Mentions',
        names='Themes',
        color_discrete_sequence=["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF"]
    )
    fig_competitors.update_traces(textinfo='percent+label', pull=[0.1, 0, 0])
    st.plotly_chart(fig_competitors, use_container_width=True)

st.markdown('<h3 class=" brand-color">Industry News</h3>', unsafe_allow_html=True)

st.markdown("**Policy News**")
reliance_tweets = [
    {"title": "Reliance launches new solar initiative in India", "url": "https://example.com/rel1", "reach": 85000, "shares": 3000},
    {"title": "Reliance partners with Aramco", "url": "https://example.com/rel2", "reach": 72000, "shares": 2500},
    {"title": "Reliance expands retail energy sector", "url": "https://example.com/rel3", "reach": 60000, "shares": 1800}
]
reliance_df = pd.DataFrame(reliance_tweets)
reliance_df['Reach (Formatted)'] = reliance_df['reach'].apply(lambda x: f"{x/1000:.0f}K" if x < 1000000 else f"{x/1000000:.1f}M")
reliance_table = reliance_df[['title', 'Reach (Formatted)', 'url', 'shares']]
reliance_table.columns = ['Snippet', 'Reach', 'Link', 'Shares']

html = "<table style='width: 100%'><tr><th>Snippet</th><th>Reach</th><th>Link</th><th>Shares</th></tr>"
for index, row in reliance_table.iterrows():
    snippet = row['Snippet']
    reach = row['Reach']
    link = row['Link']
    shares = row['Shares']
    html += f"<tr><td>{snippet}</td><td>{reach}</td><td>{link}</td><td>{shares}</td></tr>"
html += "</table>"
st.write(html, unsafe_allow_html=True)

st.markdown("**Projects & Infra**")
ntpc_tweets = [
    {"title": "NTPC achieves 100% renewable energy target", "url": "https://example.com/ntpc1", "reach": 65000, "shares": 2200},
    {"title": "NTPC signs deal for green hydrogen production", "url": "https://example.com/ntpc2", "reach": 58000, "shares": 1900},
    {"title": "NTPC expands coal-free energy portfolio", "url": "https://example.com/ntpc3", "reach": 52000, "shares": 1500}
]
ntpc_df = pd.DataFrame(ntpc_tweets)
ntpc_df['Reach (Formatted)'] = ntpc_df['reach'].apply(lambda x: f"{x/1000:.0f}K" if x < 1000000 else f"{x/1000000:.1f}M")
ntpc_table = ntpc_df[['title', 'Reach (Formatted)', 'url', 'shares']]
ntpc_table.columns = ['Snippet', 'Reach', 'Link', 'Shares']

html = "<table style='width: 100%'><tr><th>Snippet</th><th>Reach</th><th>Link</th><th>Shares</th></tr>"
for index, row in ntpc_table.iterrows():
    snippet = row['Snippet']
    reach = row['Reach']
    link = row['Link']
    shares = row['Shares']
    html += f"<tr><td>{snippet}</td><td>{reach}</td><td>{link}</td><td>{shares}</td></tr>"
html += "</table>"
st.write(html, unsafe_allow_html=True)

st.markdown("**Innovation & Tech**")
politician_tweets = [
    {"title": "Minister praises Aramco's sustainability efforts", "url": "https://example.com/pol1", "reach": 90000, "shares": 3500},
    {"title": "Policy on oil imports revised, says official", "url": "https://example.com/pol2", "reach": 78000, "shares": 2800},
    {"title": "Energy minister discusses Aramco partnership", "url": "https://example.com/pol3", "reach": 62000, "shares": 2000}
]
politician_df = pd.DataFrame(politician_tweets)
politician_df['Reach (Formatted)'] = politician_df['reach'].apply(lambda x: f"{x/1000:.0f}K" if x < 1000000 else f"{x/1000000:.1f}M")
politician_table = politician_df[['title', 'Reach (Formatted)', 'url', 'shares']]
politician_table.columns = ['Snippet', 'Reach', 'Link', 'Shares']

html = "<table style='width: 100%'><tr><th>Snippet</th><th>Reach</th><th>Link</th><th>Shares</th></tr>"
for index, row in politician_table.iterrows():
    snippet = row['Snippet']
    reach = row['Reach']
    link = row['Link']
    shares = row['Shares']
    html += f"<tr><td>{snippet}</td><td>{reach}</td><td>{link}</td><td>{shares}</td></tr>"
html += "</table>"
st.write(html, unsafe_allow_html=True)

st.markdown("**Energy Transition**")
lawyer_tweets = [
    {"title": "Legal expert comments on Aramco's new policy", "url": "https://example.com/law1", "reach": 55000, "shares": 1800},
    {"title": "Lawyer discusses Aramco's compliance strategy", "url": "https://example.com/law2", "reach": 48000, "shares": 1500},
    {"title": "Aramco faces legal scrutiny over emissions", "url": "https://example.com/law3", "reach": 42000, "shares": 1200}
]
lawyer_df = pd.DataFrame(lawyer_tweets)
lawyer_df['Reach (Formatted)'] = lawyer_df['reach'].apply(lambda x: f"{x/1000:.0f}K" if x < 1000000 else f"{x/1000000:.1f}M")
lawyer_table = lawyer_df[['title', 'Reach (Formatted)', 'url', 'shares']]
lawyer_table.columns = ['Snippet', 'Reach', 'Link', 'Shares']

html = "<table style='width: 100%'><tr><th>Snippet</th><th>Reach</th><th>Link</th><th>Shares</th></tr>"
for index, row in lawyer_table.iterrows():
    snippet = row['Snippet']
    reach = row['Reach']
    link = row['Link']
    shares = row['Shares']
    html += f"<tr><td>{snippet}</td><td>{reach}</td><td>{link}</td><td>{shares}</td></tr>"
html += "</table>"
st.write(html, unsafe_allow_html=True)

st.header('')

st.markdown('<h3 class=" brand-color">Competition Analysis</h3>', unsafe_allow_html=True)

competitor_mentions = {
    'Reliance': 35,
    'NTPC': 25,
    'Shell': 20,
    'BP': 15,
    'Chevron': 5
}

competitor_df = pd.DataFrame(list(competitor_mentions.items()), columns=['Competitor', 'Mentions'])

st.markdown("""
<style>
.card {
    border: 1px solid #e6e6e6;
    border-radius: 8px;
    padding: 15px;
    background-color: #f9f9f9;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.center-text {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    with st.container():
        st.subheader("")
        with st.expander("", expanded=True):
            st.markdown("""
            **Origins of the Content**  
            Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, *consectetur*, from a Lorem Ipsum passage, and discovered its source in classical literature.

            **Source Details**  
            Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of *De Finibus Bonorum et Malorum* (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book, a treatise on the theory of ethics, was very popular during the Renaissance. The first line of Lorem Ipsum comes from a line in section 1.10.32.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:    
    fig_competitors = px.pie(
        competitor_df,
        values='Mentions',
        names='Competitor',
        color_discrete_sequence=["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF"]
    )
    fig_competitors.update_traces(textinfo='percent+label', pull=[0.1, 0, 0])
    st.plotly_chart(fig_competitors, use_container_width=True)

st.markdown("**Reliance**")
reliance_tweets = [
    {"title": "Reliance launches new solar initiative in India", "url": "https://example.com/rel1", "reach": 85000, "shares": 3000},
    {"title": "Reliance partners with Aramco", "url": "https://example.com/rel2", "reach": 72000, "shares": 2500},
    {"title": "Reliance expands retail energy sector", "url": "https://example.com/rel3", "reach": 60000, "shares": 1800}
]
reliance_df = pd.DataFrame(reliance_tweets)
reliance_df['Reach (Formatted)'] = reliance_df['reach'].apply(lambda x: f"{x/1000:.0f}K" if x < 1000000 else f"{x/1000000:.1f}M")
reliance_table = reliance_df[['title', 'Reach (Formatted)', 'url', 'shares']]
reliance_table.columns = ['Snippet', 'Reach', 'Link', 'Shares']

html = "<table style='width: 100%'><tr><th>Snippet</th><th>Reach</th><th>Link</th><th>Shares</th></tr>"
for index, row in reliance_table.iterrows():
    snippet = row['Snippet']
    reach = row['Reach']
    link = row['Link']
    shares = row['Shares']
    html += f"<tr><td>{snippet}</td><td>{reach}</td><td>{link}</td><td>{shares}</td></tr>"
html += "</table>"
st.write(html, unsafe_allow_html=True)

# NTPC Top Tweets
st.markdown("**NTPC**")
ntpc_tweets = [
    {"title": "NTPC achieves 100% renewable energy target", "url": "https://example.com/ntpc1", "reach": 65000, "shares": 2200},
    {"title": "NTPC signs deal for green hydrogen production", "url": "https://example.com/ntpc2", "reach": 58000, "shares": 1900},
    {"title": "NTPC expands coal-free energy portfolio", "url": "https://example.com/ntpc3", "reach": 52000, "shares": 1500}
]
ntpc_df = pd.DataFrame(ntpc_tweets)
ntpc_df['Reach (Formatted)'] = ntpc_df['reach'].apply(lambda x: f"{x/1000:.0f}K" if x < 1000000 else f"{x/1000000:.1f}M")
ntpc_table = ntpc_df[['title', 'Reach (Formatted)', 'url', 'shares']]
ntpc_table.columns = ['Snippet', 'Reach', 'Link', 'Shares']

html = "<table style='width: 100%'><tr><th>Snippet</th><th>Reach</th><th>Link</th><th>Shares</th></tr>"
for index, row in ntpc_table.iterrows():
    snippet = row['Snippet']
    reach = row['Reach']
    link = row['Link']
    shares = row['Shares']
    html += f"<tr><td>{snippet}</td><td>{reach}</td><td>{link}</td><td>{shares}</td></tr>"
html += "</table>"
st.write(html, unsafe_allow_html=True)

# Politicians Top Tweets
st.markdown("**Politicians**")
politician_tweets = [
    {"title": "Minister praises Aramco's sustainability efforts", "url": "https://example.com/pol1", "reach": 90000, "shares": 3500},
    {"title": "Policy on oil imports revised, says official", "url": "https://example.com/pol2", "reach": 78000, "shares": 2800},
    {"title": "Energy minister discusses Aramco partnership", "url": "https://example.com/pol3", "reach": 62000, "shares": 2000}
]
politician_df = pd.DataFrame(politician_tweets)
politician_df['Reach (Formatted)'] = politician_df['reach'].apply(lambda x: f"{x/1000:.0f}K" if x < 1000000 else f"{x/1000000:.1f}M")
politician_table = politician_df[['title', 'Reach (Formatted)', 'url', 'shares']]
politician_table.columns = ['Snippet', 'Reach', 'Link', 'Shares']

html = "<table style='width: 100%'><tr><th>Snippet</th><th>Reach</th><th>Link</th><th>Shares</th></tr>"
for index, row in politician_table.iterrows():
    snippet = row['Snippet']
    reach = row['Reach']
    link = row['Link']
    shares = row['Shares']
    html += f"<tr><td>{snippet}</td><td>{reach}</td><td>{link}</td><td>{shares}</td></tr>"
html += "</table>"
st.write(html, unsafe_allow_html=True)

# Lawyers Top Tweets
st.markdown("**Lawyers**")
lawyer_tweets = [
    {"title": "Legal expert comments on Aramco's new policy", "url": "https://example.com/law1", "reach": 55000, "shares": 1800},
    {"title": "Lawyer discusses Aramco's compliance strategy", "url": "https://example.com/law2", "reach": 48000, "shares": 1500},
    {"title": "Aramco faces legal scrutiny over emissions", "url": "https://example.com/law3", "reach": 42000, "shares": 1200}
]
lawyer_df = pd.DataFrame(lawyer_tweets)
lawyer_df['Reach (Formatted)'] = lawyer_df['reach'].apply(lambda x: f"{x/1000:.0f}K" if x < 1000000 else f"{x/1000000:.1f}M")
lawyer_table = lawyer_df[['title', 'Reach (Formatted)', 'url', 'shares']]
lawyer_table.columns = ['Snippet', 'Reach', 'Link', 'Shares']

html = "<table style='width: 100%'><tr><th>Snippet</th><th>Reach</th><th>Link</th><th>Shares</th></tr>"
for index, row in lawyer_table.iterrows():
    snippet = row['Snippet']
    reach = row['Reach']
    link = row['Link']
    shares = row['Shares']
    html += f"<tr><td>{snippet}</td><td>{reach}</td><td>{link}</td><td>{shares}</td></tr>"
html += "</table>"
st.write(html, unsafe_allow_html=True)
st.markdown("---")
st.markdown("**Aramco India | Generated on May 23, 2025**")

