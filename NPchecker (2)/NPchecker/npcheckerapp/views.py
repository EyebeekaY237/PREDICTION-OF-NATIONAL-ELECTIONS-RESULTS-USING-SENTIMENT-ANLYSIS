from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import os
import csv
import random
from datetime import datetime, timedelta
from collections import defaultdict
from .sentiment_utils import sentiment_analyzer  # Import our enhanced analyzer

# List of available CSV files
CSV_FILES = [
    os.path.join(settings.BASE_DIR, 'data', 'sentiment1.csv'),
    os.path.join(settings.BASE_DIR, 'data', 'sentiment2.csv'),
    os.path.join(settings.BASE_DIR, 'data', 'sentiment3.csv'),
    os.path.join(settings.BASE_DIR, 'data', 'sentiment4.csv')
]

def get_random_csv_file():
    """Select a random CSV file from the available options"""
    return random.choice(CSV_FILES)

def parse_datetime(dt_str):
    """Helper to parse datetime string from CSV"""
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M'):
        try:
            return datetime.strptime(dt_str.strip(), fmt)
        except ValueError:
            continue
    return None

def get_recent_tweets(tweets, days=7):
    """Filter tweets from the last N days"""
    cutoff = datetime.now() - timedelta(days=days)
    return [t for t in tweets if parse_datetime(t['timestamp']) and parse_datetime(t['timestamp']) >= cutoff]

def extract_keywords(text):
    """Simple keyword extraction from tweet text"""
    words = text.lower().split()
    keywords = []
    for word in words:
        clean = word.strip('.,!?";:()[]')
        if (clean in ['economy', 'apc', 'pdp', 'lp', 'reform', 'manifesto',
                      'experience', 'governance', 'unity', 'youth', 'change',
                      'accountability', 'development', 'education', 'security',
                      'jobs', 'nigeria', 'vote', 'leadership']):
            keywords.append(clean.upper())
    return keywords

def get_results(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        # Select a random CSV file
        csv_file = get_random_csv_file()
        
        # Read CSV data
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        rows = []
        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        if not rows:
            raise ValueError("No data in CSV")

        # Group by candidate
        candidate_data = {'tinubu': [], 'atiku': [], 'obi': []}
        for row in rows:
            candidate = row['candidate'].strip().lower()
            if candidate in candidate_data:
                # Use ML model to analyze sentiment for more accurate results
                if 'sentiment_score' not in row or not row['sentiment_score']:
                    # Analyze with our enhanced model
                    ml_sentiment = sentiment_analyzer.analyze_sentiment(row['text'])
                    row['ml_sentiment_score'] = str(ml_sentiment)
                else:
                    # Use existing score but also compute ML score for comparison
                    try:
                        original_score = float(row['sentiment_score'])
                        row['ml_sentiment_score'] = str(sentiment_analyzer.analyze_sentiment(row['text']))
                    except ValueError:
                        row['ml_sentiment_score'] = str(sentiment_analyzer.analyze_sentiment(row['text']))
                
                candidate_data[candidate].append(row)

        # Analyze each candidate
        results = {}
        total_votes = 20_000_000
        sentiment_values = []
        ml_sentiment_values = []

        for candidate in ['tinubu', 'atiku', 'obi']:
            data = candidate_data[candidate]
            if not data:
                continue

            # Filter recent tweets (last 7 days)
            recent_tweets = get_recent_tweets(data, days=7)
            sample = random.sample(recent_tweets, min(100, len(recent_tweets))) if recent_tweets else data

            # Calculate average sentiment using both original and ML scores
            sentiment_scores = []
            ml_sentiment_scores = []
            
            for row in sample:
                try:
                    # Use original score if available, otherwise use ML score
                    if 'sentiment_score' in row and row['sentiment_score']:
                        score = float(row['sentiment_score'])
                    else:
                        score = float(row.get('ml_sentiment_score', 0))
                    sentiment_scores.append(score)
                except (ValueError, KeyError):
                    # Fallback to ML analysis
                    ml_score = sentiment_analyzer.analyze_sentiment(row['text'])
                    sentiment_scores.append(ml_score)
                
                # Always track ML sentiment separately
                try:
                    ml_score = float(row.get('ml_sentiment_score', 
                                           str(sentiment_analyzer.analyze_sentiment(row['text']))))
                    ml_sentiment_scores.append(ml_score)
                except (ValueError, KeyError):
                    ml_score = sentiment_analyzer.analyze_sentiment(row['text'])
                    ml_sentiment_scores.append(ml_score)
            
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            avg_ml_sentiment = sum(ml_sentiment_scores) / len(ml_sentiment_scores) if ml_sentiment_scores else 0

            # Engagement stats
            likes = [int(row.get('likes', 0)) for row in sample]
            retweets = [int(row.get('retweets', 0)) for row in sample]
            total_reach = sum(likes) + sum(retweets) * 2

            # Extract top keywords from sampled tweets
            all_keywords = []
            for row in sample:
                all_keywords.extend(extract_keywords(row['text']))
            top_keywords = list(set(all_keywords))[:4]
            if len(top_keywords) < 4:
                fallback = {
                    'tinubu': ['ECONOMY', 'APC', 'REFORM', 'MANIFESTO'],
                    'atiku': ['EXPERIENCE', 'PDP', 'GOVERNANCE', 'UNITY'],
                    'obi': ['YOUTH', 'LP', 'CHANGE', 'ACCOUNTABILITY']
                }
                top_keywords = (top_keywords + fallback[candidate])[:4]

            # Store candidate results
            results[candidate] = {
                'votes': 0,
                'percentage': 0,
                'sentiment': round(avg_sentiment, 4),
                'ml_sentiment': round(avg_ml_sentiment, 4)  # Add ML sentiment
            }

            # Store detailed analysis
            results.setdefault('detailed', {})[candidate] = {
                'tweet_count': len(sample),
                'top_keywords': top_keywords,
                'engagement': {
                    'avg_likes': int(sum(likes) / len(likes)) if likes else 0,
                    'avg_retweets': int(sum(retweets) / len(retweets)) if retweets else 0,
                    'total_reach': total_reach
                },
                'sentiment_analysis': {
                    'original_avg': round(avg_sentiment, 4),
                    'ml_avg': round(avg_ml_sentiment, 4),
                    'sample_size': len(sample)
                }
            }
            sentiment_values.append(avg_sentiment)
            ml_sentiment_values.append(avg_ml_sentiment)

        # Use ML sentiment for more accurate percentage calculations
        total_strength = sum(abs(sv) for sv in ml_sentiment_values)
        if total_strength == 0:
            # Fallback to original sentiment if ML fails
            total_strength = sum(abs(sv) for sv in sentiment_values)
            if total_strength == 0:
                tinubu_percent = 40.0
                atiku_percent = 35.0
                obi_percent = 25.0
            else:
                percentages = [(abs(sv) / total_strength) * 100 for sv in sentiment_values]
                tinubu_percent = percentages[0] if len(percentages) > 0 else 40.0
                atiku_percent = percentages[1] if len(percentages) > 1 else 35.0
                obi_percent = percentages[2] if len(percentages) > 2 else 25.0
        else:
            # Use ML sentiment for calculation
            percentages = [(abs(sv) / total_strength) * 100 for sv in ml_sentiment_values]
            tinubu_percent = percentages[0] if len(percentages) > 0 else 40.0
            atiku_percent = percentages[1] if len(percentages) > 1 else 35.0
            obi_percent = percentages[2] if len(percentages) > 2 else 25.0

        # Assign final percentages and votes
        results['tinubu']['percentage'] = round(tinubu_percent, 2)
        results['atiku']['percentage'] = round(atiku_percent, 2)
        results['obi']['percentage'] = round(obi_percent, 2)

        results['tinubu']['votes'] = int(total_votes * (tinubu_percent / 100))
        results['atiku']['votes'] = int(total_votes * (atiku_percent / 100))
        results['obi']['votes'] = int(total_votes * (obi_percent / 100))

        # Add model info to response
        results['model_info'] = {
            'using_ml_model': sentiment_analyzer.model is not None,
            'analysis_type': 'ML Enhanced' if sentiment_analyzer.model else 'Rule-Based Fallback',
            'timestamp': datetime.now().isoformat(),
            'data_source': os.path.basename(csv_file)  # Add which CSV file was used
        }

        return JsonResponse(results, safe=False)

    except FileNotFoundError as e:
        return JsonResponse({'error': 'Sentiment data file not found. Please check configuration.'}, status=500)
    except Exception as e:
        return JsonResponse({'error': f'Internal server error: {str(e)}'}, status=500)


def home(request):
    """Render the home page with initial data"""
    # Use the sentiment analyzer to get real-time sentiment
    sample_tinubu = "Tinubu is building a strong economy for Nigeria"
    sample_atiku = "Atiku has the experience to fix Nigeria's economy"
    sample_obi = "Peter Obi is the voice of the Nigerian youth"
    
    tinubu_sentiment = sentiment_analyzer.analyze_sentiment(sample_tinubu)
    atiku_sentiment = sentiment_analyzer.analyze_sentiment(sample_atiku)
    obi_sentiment = sentiment_analyzer.analyze_sentiment(sample_obi)
    
    # Calculate percentages based on ML sentiment
    sentiments = [abs(tinubu_sentiment), abs(atiku_sentiment), abs(obi_sentiment)]
    total = sum(sentiments) if sum(sentiments) > 0 else 1
    
    tinubu_percent = (abs(tinubu_sentiment) / total) * 100
    atiku_percent = (abs(atiku_sentiment) / total) * 100
    obi_percent = (abs(obi_sentiment) / total) * 100

    context = {
        'candidates': [
            {
                'name': 'Bola Tinubu',
                'party': 'APC',
                'image': 'https://www.wathi.org/wp-content/uploads/2023/02/Bola-Ahmed-Tinubu.jpg',
                'twitter_handle': '@officialABAT',
                'percentage': round(tinubu_percent, 2),
                'sentiment': round(tinubu_sentiment, 2),
                'votes': int(20_000_000 * (tinubu_percent / 100))
            },
            {
                'name': 'Atiku Abubakar',
                'party': 'PDP',
                'image': 'https://th.bing.com/th/id/OIP.yppVgq-WyNXtLT4q0aXFcwAAAA?rs=1&pid=ImgDetMain',
                'twitter_handle': '@atiku',
                'percentage': round(atiku_percent, 2),
                'sentiment': round(atiku_sentiment, 2),
                'votes': int(20_000_000 * (atiku_percent / 100))
            },
            {
                'name': 'Peter Obi',
                'party': 'LP',
                'image': 'https://th.bing.com/th/id/OIP.nQNXC6hoOjrWv8oYMWSACgHaEU?rs=1&pid=ImgDetMain',
                'twitter_handle': '@PeterObi',
                'percentage': round(obi_percent, 2),
                'sentiment': round(obi_sentiment, 2),
                'votes': int(20_000_000 * (obi_percent / 100))
            }
        ],
        'initial_data': {
            'tinubu': {
                'votes': int(20_000_000 * (tinubu_percent / 100)),
                'percentage': round(tinubu_percent, 2),
                'sentiment': round(tinubu_sentiment, 2)
            },
            'atiku': {
                'votes': int(20_000_000 * (atiku_percent / 100)),
                'percentage': round(atiku_percent, 2),
                'sentiment': round(atiku_sentiment, 2)
            },
            'obi': {
                'votes': int(20_000_000 * (obi_percent / 100)),
                'percentage': round(obi_percent, 2),
                'sentiment': round(obi_sentiment, 2)
            }
        },
        'model_status': 'ML Model Active' if sentiment_analyzer.model else 'Rule-Based Analysis'
    }

    return render(request, 'index.html', context)


def analyze_single_tweet(request):
    """API endpoint to analyze single tweet sentiment"""
    if request.method == 'POST':
        try:
            data = request.POST
            tweet_text = data.get('text', '')
            
            if not tweet_text:
                return JsonResponse({'error': 'No text provided'}, status=400)
            
            # Analyze sentiment using our model
            sentiment_score = sentiment_analyzer.analyze_sentiment(tweet_text)
            
            return JsonResponse({
                'text': tweet_text,
                'sentiment_score': round(sentiment_score, 4),
                'sentiment_label': 'Positive' if sentiment_score > 0 else 'Negative' if sentiment_score < 0 else 'Neutral',
                'analysis_type': 'ML Model' if sentiment_analyzer.model else 'Rule-Based'
            })
            
        except Exception as e:
            return JsonResponse({'error': f'Analysis failed: {str(e)}'}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


def get_twitter_sentiment(candidate_queries, count=100):
    """Fetch and analyze tweets for each candidate with fallback"""
    candidate_sentiments = defaultdict(list)
    
    # Use our enhanced sentiment analyzer instead of fallback
    fallback_sentiments = {
        'tinubu': [0.5, 0.6, 0.4, 0.7, 0.3],
        'atiku': [0.3, 0.4, 0.2, 0.5, 0.6],
        'obi': [0.7, 0.8, 0.6, 0.9, 0.5]
    }
    
    try:
        # Try to get real data
        api_key = os.getenv('TWITTER_API_KEY')
        if not api_key:
            # Use our analyzer on fallback data
            for candidate, sentiments in fallback_sentiments.items():
                candidate_sentiments[candidate] = [sentiment_analyzer.analyze_sentiment(
                    f"{candidate} is doing well" if s > 0 else f"{candidate} is not good"
                ) for s in sentiments]
            return candidate_sentiments
            
        # ... rest of your Twitter API code ...
        # Replace analyze_sentiment() calls with sentiment_analyzer.analyze_sentiment()
        
    except Exception as e:
        print(f"Twitter API failed, using enhanced analysis: {str(e)}")
        for candidate, sentiments in fallback_sentiments.items():
            candidate_sentiments[candidate] = [sentiment_analyzer.analyze_sentiment(
                f"{candidate} positive tweet" if s > 0 else f"{candidate} negative tweet"
            ) for s in sentiments]
        return candidate_sentiments


def history(request): 
    """Single history function implementation"""
    elections = [
        {
            'year': '1999',
            'date': 'February 27, 1999',
            'winner': 'Olusegun Obasanjo',
            'party': 'PDP',
            'percentage': '62.8%',
            'significance': 'Return to civilian rule after military regime'
        },
        # Add other elections similarly
    ]
    
    return render(request, 'history.html', {
        'elections': elections,
        'stats': {
            'first_election': 'December 12, 1959',
            'total_elections': 12,
            'highest_turnout': '69.1% (1993)'
        }
    })
 
def search(request):
    return render(request, 'search.html')

def login(request):
    candidates = [
        {'name': 'Candidate A', 'party': 'APC', 'image': 'https://www.wathi.org/wp-content/uploads/2023/02/Bola-Ahmed-Tinubu.jpg'},
        {'name': 'Candidate B', 'party': 'PDP', 'image': 'candidate2.jpg'},
        {'name': 'Candidate C', 'party': 'LP', 'image': 'candidate3.jpg'},
    ]
    return render(request, 'index.html', {'candidates': candidates})

def results(request):
    return render(request, 'results.html')

def about(request):
    return render(request, 'about.html',{"imagedata":"media/20240826_072029.jpg"})

def coat_of_arm(request):
    return render(request, 'coat_of_arm.html')