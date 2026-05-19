import json

bl = json.load(open('simulation/metrics_history/baseline_metrics.json'))
print('YUE BASELINE keys:', list(bl.keys()))
print('community_health:', bl.get('community_health', {}))
print('coverage:', bl.get('coverage', {}))
print('user_activity_fit:', bl.get('user_activity_fit', {}))
