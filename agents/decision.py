class DecisionAgent:
    def decide(self, topic, summary):
        if topic == 'Security':
            return 'trigger_alert', {'message': f"Security alert: {summary}"}
        elif topic == 'Finance':
            return 'log_and_notify', {'message': summary}
        elif topic == 'Technology':
            return 'update_dashboard', {'data': summary}
        else:
            return 'log', {'message': summary}