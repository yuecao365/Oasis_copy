#!/usr/bin/env python3
"""
Generate HTML visualizations for user rankings and personas.
Creates an interactive HTML dashboard showing:
1. Top users by category/topic
2. User rankings across topics
3. Sample personas and their characteristics
4. Category distribution
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from config import OUTPUT_DIR as CONFIG_OUTPUT_DIR, VISUALIZATIONS_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RANKINGS_FILE = CONFIG_OUTPUT_DIR / "user_rankings.json"
PERSONAS_FILE = CONFIG_OUTPUT_DIR / "user_personas.json"
HTML_FILE = VISUALIZATIONS_DIR / "rankings_dashboard.html"


def load_data():
    """Load rankings and personas data."""
    with RANKINGS_FILE.open() as f:
        rankings = json.load(f)
    
    with PERSONAS_FILE.open() as f:
        personas = json.load(f)
    
    return rankings, personas


def generate_html_dashboard(rankings: List[Dict], personas: List[Dict]):
    """Generate complete HTML dashboard."""
    
    # Calculate statistics
    total_categories = len(rankings)
    total_personas = len(personas)
    total_users_ranked = sum(len(c.get('top_users', [])) for c in rankings)
    
    # Top categories
    top_categories = sorted([(c.get('topic', f'Topic {c.get("cluster_id")}'), 
                            len(c.get('top_users', []))) 
                           for c in rankings], 
                          key=lambda x: x[1], reverse=True)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>User Rankings & Personas Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 30px;
        }}
        
        h1 {{
            color: #667eea;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .section {{
            margin-bottom: 50px;
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .category-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
        }}
        
        .category-card {{
            background: #f8f9fa;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        
        .category-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            border-color: #667eea;
        }}
        
        .category-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .category-name {{
            font-size: 1.3em;
            font-weight: bold;
            color: #667eea;
            flex: 1;
        }}
        
        .user-count-badge {{
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }}
        
        .user-list {{
            list-style: none;
        }}
        
        .user-item {{
            background: white;
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .user-info {{
            flex: 1;
        }}
        
        .user-name {{
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }}
        
        .user-stats {{
            font-size: 0.85em;
            color: #666;
        }}
        
        .tweet-count {{
            background: #764ba2;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        
        .persona-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
        }}
        
        .persona-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        
        .persona-card:hover {{
            transform: scale(1.02);
        }}
        
        .persona-header {{
            margin-bottom: 15px;
        }}
        
        .persona-name {{
            font-size: 1.4em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        
        .persona-username {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .persona-category {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.85em;
            margin-top: 10px;
        }}
        
        .persona-bio {{
            font-style: italic;
            color: #555;
            margin: 15px 0;
            padding: 10px;
            background: white;
            border-radius: 5px;
        }}
        
        .persona-details {{
            margin-top: 15px;
        }}
        
        .detail-item {{
            margin: 8px 0;
            font-size: 0.9em;
        }}
        
        .detail-label {{
            font-weight: bold;
            color: #667eea;
        }}
        
        .sample-tweet {{
            background: white;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 0.85em;
            color: #555;
            border-left: 3px solid #667eea;
        }}
        
        .top-categories-list {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }}
        
        .category-item {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #667eea;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .category-item-name {{
            font-weight: bold;
            color: #333;
        }}
        
        .category-item-count {{
            background: #667eea;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 User Rankings & Personas Dashboard</h1>
        <p class="subtitle">Analysis of Top Users by Category and Generated Personas</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{total_categories}</div>
                <div class="stat-label">Categories</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{total_personas}</div>
                <div class="stat-label">Personas Generated</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{total_users_ranked}</div>
                <div class="stat-label">Users Ranked</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{total_users_ranked // total_categories if total_categories > 0 else 0}</div>
                <div class="stat-label">Avg Users/Category</div>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">🏆 Top Categories by User Count</h2>
            <div class="top-categories-list">
"""
    
    for topic, count in top_categories[:10]:
        html += f"""
                <div class="category-item">
                    <div class="category-item-name">{topic[:50]}{'...' if len(topic) > 50 else ''}</div>
                    <div class="category-item-count">{count} users</div>
                </div>
"""
    
    html += """
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">👥 Top Users by Category</h2>
            <div class="category-grid">
"""
    
    for cluster in rankings:
        topic = cluster.get('topic', f'Topic {cluster.get("cluster_id", 0)}')
        top_users = cluster.get('top_users', [])[:10]
        
        html += f"""
                <div class="category-card">
                    <div class="category-header">
                        <div class="category-name">{topic}</div>
                        <div class="user-count-badge">{len(top_users)} users</div>
                    </div>
                    <ul class="user-list">
"""
        
        for i, user in enumerate(top_users, 1):
            user_name = user.get('user_name', 'Unknown')
            tweet_count = user.get('tweet_count', 0)
            sample_tweets = user.get('sample_tweets', [])
            sample = sample_tweets[0] if sample_tweets else "No sample tweets"
            if len(sample) > 100:
                sample = sample[:100] + "..."
            
            html += f"""
                        <li class="user-item">
                            <div class="user-info">
                                <div class="user-name">#{i} @{user_name}</div>
                                <div class="user-stats">📊 {tweet_count} tweets</div>
                                <div class="sample-tweet">"{sample}"</div>
                            </div>
                            <div class="tweet-count">{tweet_count}</div>
                        </li>
"""
        
        html += """
                    </ul>
                </div>
"""
    
    html += """
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">🎭 Generated Personas</h2>
            <div class="persona-grid">
"""
    
    for persona in personas:
        # Always display the ORIGINAL username from the data.
        handle = persona.get('original_user_name') or persona.get('username', 'unknown')
        display_name = handle
        bio = persona.get('bio', 'No bio available')
        persona_desc = persona.get('persona', 'No persona description')
        category = persona.get('category', 'Unknown')
        age = persona.get('age', 'N/A')
        gender = persona.get('gender', 'N/A')
        mbti = persona.get('mbti', 'N/A')
        country = persona.get('country', 'N/A')
        profession = persona.get('profession', 'N/A')
        tweet_count = persona.get('tweet_count', 0)
        
        html += f"""
                <div class="persona-card">
                    <div class="persona-header">
                        <div class="persona-name">@{display_name}</div>
                        <div class="persona-category">{category}</div>
                    </div>
                    <div class="persona-bio">"{bio}"</div>
                    <div class="persona-details">
                        <div class="detail-item">
                            <span class="detail-label">Age:</span> {age} | 
                            <span class="detail-label">Gender:</span> {gender} | 
                            <span class="detail-label">MBTI:</span> {mbti}
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Country:</span> {country} | 
                            <span class="detail-label">Profession:</span> {profession}
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Tweet Count:</span> {tweet_count} tweets in category
                        </div>
                    </div>
                    <div class="sample-tweet">
                        <strong>Persona:</strong> {persona_desc[:200]}{'...' if len(persona_desc) > 200 else ''}
                    </div>
                </div>
"""
    
    html += """
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    with HTML_FILE.open('w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ HTML dashboard saved to: {HTML_FILE}")
    print(f"   Open in browser: file://{HTML_FILE.absolute()}")


def main():
    """Main function."""
    print("Loading data...")
    rankings, personas = load_data()
    
    print(f"Generating HTML dashboard...")
    print(f"  • {len(rankings)} categories")
    print(f"  • {len(personas)} personas")
    
    generate_html_dashboard(rankings, personas)
    
    print(f"\n✅ Dashboard generated successfully!")
    print(f"\nOpen the file in your browser:")
    print(f"   {HTML_FILE.absolute()}")


if __name__ == "__main__":
    main()
