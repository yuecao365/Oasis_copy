#!/usr/bin/env python3
"""
Visualize user rankings and personas from the persona generation pipeline.
Creates visualizations showing:
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

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not installed. Install with: pip install matplotlib seaborn")

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from config import OUTPUT_DIR as CONFIG_OUTPUT_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RANKINGS_FILE = CONFIG_OUTPUT_DIR / "user_rankings.json"
PERSONAS_FILE = CONFIG_OUTPUT_DIR / "user_personas.json"
OUTPUT_DIR = CONFIG_OUTPUT_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load rankings and personas data."""
    with RANKINGS_FILE.open() as f:
        rankings = json.load(f)
    
    with PERSONAS_FILE.open() as f:
        personas = json.load(f)
    
    return rankings, personas


def create_top_users_by_category(rankings: List[Dict], top_n: int = 10):
    """Create bar chart showing top users by tweet count for each category."""
    if not HAS_PLOTTING:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Top Users by Category (Ranked by Tweet Count)', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, cluster in enumerate(rankings[:4]):  # Show top 4 categories
        topic = cluster.get('topic', f'Topic {cluster.get("cluster_id", idx)}')
        top_users = cluster.get('top_users', [])[:top_n]
        
        if not top_users:
            continue
        
        users = [u['user_name'][:20] for u in top_users]  # Truncate long names
        counts = [u['tweet_count'] for u in top_users]
        
        ax = axes[idx]
        bars = ax.barh(range(len(users)), counts, color=plt.cm.viridis(range(len(users))))
        ax.set_yticks(range(len(users)))
        ax.set_yticklabels(users, fontsize=9)
        ax.set_xlabel('Tweet Count', fontsize=10)
        ax.set_title(f'{topic[:40]}...' if len(topic) > 40 else topic, fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count + 0.5, i, str(count), va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top_users_by_category.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'top_users_by_category.png'}")
    plt.close()


def create_category_distribution(rankings: List[Dict]):
    """Create pie chart showing distribution of users across categories."""
    if not HAS_PLOTTING:
        return
    
    category_counts = {}
    for cluster in rankings:
        topic = cluster.get('topic', f'Topic {cluster.get("cluster_id", 0)}')
        user_count = len(cluster.get('top_users', []))
        category_counts[topic] = user_count
    
    # Sort by count and take top 10
    sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    labels = [cat[:30] + '...' if len(cat) > 30 else cat for cat, _ in sorted_cats]
    sizes = [count for _, count in sorted_cats]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.Set3(range(len(labels)))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                      colors=colors, startangle=90)
    
    # Improve text readability
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    ax.set_title('Distribution of Top Users Across Categories', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'category_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'category_distribution.png'}")
    plt.close()


def create_user_rankings_heatmap(rankings: List[Dict], top_users_count: int = 20):
    """Create heatmap showing which users rank highly in which categories."""
    if not HAS_PLOTTING:
        return
    
    # Collect all unique users and their rankings across categories
    user_rankings = defaultdict(dict)
    category_names = []
    
    for cluster in rankings:
        topic = cluster.get('topic', f'Topic {cluster.get("cluster_id", 0)}')
        category_names.append(topic)
        top_users = cluster.get('top_users', [])
        
        for rank, user in enumerate(top_users[:top_users_count], 1):
            user_id = user['user_name']
            user_rankings[user_id][topic] = rank
    
    # Get top users who appear in multiple categories
    multi_category_users = {uid: data for uid, data in user_rankings.items() if len(data) > 1}
    if not multi_category_users:
        print("No users appear in multiple categories for heatmap")
        return
    
    # Sort by number of categories they appear in
    sorted_users = sorted(multi_category_users.items(), key=lambda x: len(x[1]), reverse=True)[:15]
    
    # Build matrix
    user_names = [uid for uid, _ in sorted_users]
    matrix = []
    for user_id, _ in sorted_users:
        row = []
        for topic in category_names[:10]:  # Limit to top 10 categories
            rank = user_rankings[user_id].get(topic)
            row.append(rank if rank else None)
        matrix.append(row)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(range(len(category_names[:10])))
    ax.set_xticklabels([cat[:25] + '...' if len(cat) > 25 else cat for cat in category_names[:10]], 
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(user_names)))
    ax.set_yticklabels([name[:20] for name in user_names], fontsize=8)
    
    # Add text annotations
    for i in range(len(user_names)):
        for j in range(len(category_names[:10])):
            if matrix[i][j] is not None:
                text = ax.text(j, i, str(matrix[i][j]), ha="center", va="center", 
                              color="black", fontsize=7, fontweight='bold')
    
    ax.set_title('User Rankings Across Categories\n(Lower rank = better, blank = not ranked)', 
                fontsize=12, fontweight='bold', pad=15)
    plt.colorbar(im, ax=ax, label='Rank (lower is better)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'user_rankings_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'user_rankings_heatmap.png'}")
    plt.close()


def create_persona_examples(personas: List[Dict], num_examples: int = 6):
    """Create visual cards showing example personas."""
    if not HAS_PLOTTING:
        return
    
    # Select diverse personas (different categories)
    selected = []
    seen_categories = set()
    for persona in personas:
        category = persona.get('category', '')
        if category not in seen_categories and len(selected) < num_examples:
            selected.append(persona)
            seen_categories.add(category)
    
    if len(selected) < num_examples:
        # Fill remaining slots
        for persona in personas:
            if persona not in selected and len(selected) < num_examples:
                selected.append(persona)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Example User Personas', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, persona in enumerate(selected[:6]):
        ax = axes[idx]
        ax.axis('off')
        
        # Create persona card
        username = persona.get('username', 'Unknown')
        realname = persona.get('realname', 'Unknown')
        bio = persona.get('bio', 'No bio available')
        persona_desc = persona.get('persona', 'No persona description')
        category = persona.get('category', 'Unknown')
        age = persona.get('age', 'N/A')
        gender = persona.get('gender', 'N/A')
        mbti = persona.get('mbti', 'N/A')
        country = persona.get('country', 'N/A')
        profession = persona.get('profession', 'N/A')
        tweet_count = persona.get('tweet_count', 0)
        
        # Card background
        card = Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                        facecolor='#f0f0f0', edgecolor='#333', linewidth=2)
        ax.add_patch(card)
        
        # Title
        ax.text(0.5, 0.95, f'{realname} (@{username})', 
               transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Category badge
        ax.text(0.5, 0.88, f'🏆 {category}', 
               transform=ax.transAxes, ha='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='#4CAF50', alpha=0.7))
        
        # Demographics
        demo_text = f"Age: {age} | {gender.title()} | {mbti} | {country}"
        ax.text(0.5, 0.80, demo_text, transform=ax.transAxes, ha='center', fontsize=9)
        
        # Profession
        ax.text(0.5, 0.75, f"💼 {profession}", transform=ax.transAxes, ha='center', fontsize=9)
        
        # Bio
        ax.text(0.5, 0.68, f'"{bio}"', transform=ax.transAxes, ha='center', 
               fontsize=8, style='italic', wrap=True)
        
        # Persona description (truncated)
        desc_lines = []
        words = persona_desc.split()
        current_line = []
        for word in words[:40]:  # Limit to ~40 words
            current_line.append(word)
            if len(' '.join(current_line)) > 60:
                desc_lines.append(' '.join(current_line[:-1]))
                current_line = [word]
        if current_line:
            desc_lines.append(' '.join(current_line))
        
        y_pos = 0.60
        for line in desc_lines[:4]:  # Max 4 lines
            ax.text(0.5, y_pos, line, transform=ax.transAxes, ha='center', 
                   fontsize=7, wrap=True)
            y_pos -= 0.05
        
        # Tweet count
        ax.text(0.5, 0.05, f'📊 {tweet_count} tweets in category', 
               transform=ax.transAxes, ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'persona_examples.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'persona_examples.png'}")
    plt.close()


def create_summary_statistics(rankings: List[Dict], personas: List[Dict]):
    """Create a summary statistics visualization."""
    if not HAS_PLOTTING:
        return
    
    # Calculate statistics
    total_categories = len(rankings)
    total_personas = len(personas)
    total_users_ranked = sum(len(c.get('top_users', [])) for c in rankings)
    
    # Average tweets per user per category
    avg_tweets = []
    for cluster in rankings:
        users = cluster.get('top_users', [])
        if users:
            avg_tweets.append(sum(u['tweet_count'] for u in users) / len(users))
    
    # Category sizes
    category_sizes = [len(c.get('top_users', [])) for c in rankings]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('User Rankings & Personas - Summary Statistics', fontsize=16, fontweight='bold')
    
    # 1. Key metrics
    ax1 = axes[0, 0]
    ax1.axis('off')
    metrics_text = f"""
    📊 Dataset Summary
    
    Total Categories: {total_categories}
    Total Personas Generated: {total_personas}
    Total Users Ranked: {total_users_ranked}
    
    Average Users per Category: {total_users_ranked / total_categories:.1f}
    Average Tweets per User: {sum(avg_tweets) / len(avg_tweets) if avg_tweets else 0:.1f}
    """
    ax1.text(0.5, 0.5, metrics_text, transform=ax1.transAxes, 
            ha='center', va='center', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 2. Category size distribution
    ax2 = axes[0, 1]
    ax2.hist(category_sizes, bins=min(20, len(set(category_sizes))), 
            color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Users per Category', fontsize=10)
    ax2.set_ylabel('Number of Categories', fontsize=10)
    ax2.set_title('Distribution of Category Sizes', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Top categories by user count
    ax3 = axes[1, 0]
    top_categories = sorted([(c.get('topic', f'Topic {c.get("cluster_id")}'), 
                            len(c.get('top_users', []))) 
                           for c in rankings], 
                          key=lambda x: x[1], reverse=True)[:10]
    categories = [cat[:30] + '...' if len(cat) > 30 else cat for cat, _ in top_categories]
    counts = [cnt for _, cnt in top_categories]
    ax3.barh(range(len(categories)), counts, color='coral')
    ax3.set_yticks(range(len(categories)))
    ax3.set_yticklabels(categories, fontsize=9)
    ax3.set_xlabel('Number of Ranked Users', fontsize=10)
    ax3.set_title('Top 10 Categories by User Count', fontsize=11, fontweight='bold')
    ax3.invert_yaxis()
    
    # 4. Average tweets per category
    ax4 = axes[1, 1]
    category_avg_tweets = []
    category_names_short = []
    for cluster in rankings[:10]:  # Top 10
        topic = cluster.get('topic', f'Topic {cluster.get("cluster_id")}')
        users = cluster.get('top_users', [])
        if users:
            avg = sum(u['tweet_count'] for u in users) / len(users)
            category_avg_tweets.append(avg)
            category_names_short.append(topic[:25] + '...' if len(topic) > 25 else topic)
    
    ax4.bar(range(len(category_avg_tweets)), category_avg_tweets, color='lightgreen')
    ax4.set_xticks(range(len(category_names_short)))
    ax4.set_xticklabels(category_names_short, rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Average Tweets per User', fontsize=10)
    ax4.set_title('Average Tweet Count by Category', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'summary_statistics.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'summary_statistics.png'}")
    plt.close()


def print_text_summary(rankings: List[Dict], personas: List[Dict]):
    """Print a text summary to console."""
    print("\n" + "="*80)
    print("USER RANKINGS & PERSONAS SUMMARY")
    print("="*80)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  • Total Categories: {len(rankings)}")
    print(f"  • Total Personas Generated: {len(personas)}")
    print(f"  • Total Users Ranked: {sum(len(c.get('top_users', [])) for c in rankings)}")
    
    print(f"\n🏆 Top Categories by User Count:")
    top_cats = sorted([(c.get('topic', f'Topic {c.get("cluster_id")}'), 
                       len(c.get('top_users', []))) 
                      for c in rankings], 
                     key=lambda x: x[1], reverse=True)[:5]
    for i, (topic, count) in enumerate(top_cats, 1):
        print(f"  {i}. {topic}: {count} users")
    
    print(f"\n👤 Example Top Users by Category:")
    for cluster in rankings[:3]:
        topic = cluster.get('topic', f'Topic {cluster.get("cluster_id")}')
        print(f"\n  Category: {topic}")
        top_users = cluster.get('top_users', [])[:3]
        for i, user in enumerate(top_users, 1):
            print(f"    {i}. @{user['user_name']}: {user['tweet_count']} tweets")
            if user.get('sample_tweets'):
                sample = user['sample_tweets'][0][:60] + '...' if len(user['sample_tweets'][0]) > 60 else user['sample_tweets'][0]
                print(f"       Sample: \"{sample}\"")
    
    print(f"\n🎭 Example Personas:")
    for persona in personas[:3]:
        print(f"\n  • {persona.get('realname', 'Unknown')} (@{persona.get('username', 'unknown')})")
        print(f"    Category: {persona.get('category', 'Unknown')}")
        print(f"    Bio: {persona.get('bio', 'No bio')}")
        print(f"    Persona: {persona.get('persona', 'No description')[:150]}...")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main function to generate all visualizations."""
    print("Loading data...")
    rankings, personas = load_data()
    
    if not HAS_PLOTTING:
        print("\n⚠️  matplotlib/seaborn not available. Generating text summary only.")
        print_text_summary(rankings, personas)
        return
    
    print(f"\nGenerating visualizations...")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Create visualizations
    print("1. Creating top users by category chart...")
    create_top_users_by_category(rankings)
    
    print("2. Creating category distribution chart...")
    create_category_distribution(rankings)
    
    print("3. Creating user rankings heatmap...")
    create_user_rankings_heatmap(rankings)
    
    print("4. Creating persona examples...")
    create_persona_examples(personas)
    
    print("5. Creating summary statistics...")
    create_summary_statistics(rankings, personas)
    
    # Also print text summary
    print_text_summary(rankings, personas)
    
    print(f"\n✅ All visualizations saved to: {OUTPUT_DIR}")
    print(f"\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  • {file.name}")


if __name__ == "__main__":
    main()
