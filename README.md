# Ade Webhook Market Monitor

Webhook-only Coinbase listing monitor.

## Required secrets

WEBHOOK
WEBHOOK2 (optional)

## Run

pip install -r requirements.txt
python bot.py

On first run, the script creates a baseline of current Coinbase assets
and does not send historical listing alerts. New assets detected after
that baseline trigger Discord webhook embeds.
