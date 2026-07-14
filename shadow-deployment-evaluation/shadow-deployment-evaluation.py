import math
def evaluate_shadow(production_log, shadow_log, criteria):
    """
    Evaluate whether a shadow model is ready for promotion.
    """
    total_prod_pred = len(production_log)
    total_correct_prod_pred = sum((x['actual'] == x['prediction']) 
                                  for x in production_log)
    total_correct_sh_pred = sum((x['actual'] == x['prediction']) 
                                for x in shadow_log)
    total_shadow_pred = len(shadow_log)

    shadow_acc = total_correct_sh_pred/total_shadow_pred
    prod_acc = total_correct_prod_pred/total_prod_pred

    acc_gain = shadow_acc - prod_acc
    sorted_latency = sorted(shadow_log, key = lambda x: x['latency_ms'])
    rank = 0.95*len(sorted_latency)
    idx = math.ceil(rank) - 1
    shadow_log_p95 = sorted_latency[idx]['latency_ms']
    sh_prod_pred = 0
    for sh,prod in zip(shadow_log,production_log):
        if sh['prediction'] == prod['prediction']:
            sh_prod_pred+=1

    agreement_rate = sh_prod_pred/len(production_log)

    is_promoted = (acc_gain >= criteria['min_accuracy_gain']
                  and shadow_log_p95 <= criteria['max_latency_p95']
                  and agreement_rate >= criteria['min_agreement_rate'])

    metrics = {
        'shadow_accuracy': shadow_acc,
        'production_accuracy': prod_acc,
        'accuracy_gain': acc_gain,
        'shadow_latency_p95': shadow_log_p95,
        'agreement_rate': agreement_rate
    }

    return {'promote':is_promoted, 'metrics':metrics}