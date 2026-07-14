def retraining_policy(daily_stats, config):
    """
    Decide which days to trigger model retraining.
    """
    budget = config["budget"]
    retrain_days = []

    last_retrain_day = 1 - config["cooldown"]

    for day in daily_stats:
        days_since_retrain = day["day"] - last_retrain_day

        retrain_needed = (
            day["drift_score"] > config["drift_threshold"]
            or day["performance"] < config["performance_threshold"]
            or days_since_retrain >= config["max_staleness"]
        )

        cooldown_ok = days_since_retrain >= config["cooldown"]
        budget_ok = budget >= config["retrain_cost"]

        if retrain_needed and cooldown_ok and budget_ok:
            retrain_days.append(day["day"])
            budget -= config["retrain_cost"]
            last_retrain_day = day["day"]

    return retrain_days