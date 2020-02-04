from cryptolytic.data.historical import live_update
from cryptolytic.model.cron_model import xgb_cron_train, xgb_cron_pred
live_update()
xgb_cron+train('trade')
xgb_cron_train('arbitrage')
xgb_cron_pred('trade')
xgb_cron_pred('arbitrage')
# Crontab: 5 * * * * python3 /var/www/(ipv4 address)/cronscript.py
