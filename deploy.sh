#!/bin/sh

# echo "--- MIGRATE TABLES ---"

# python3 migrate.py

# echo "--- MIGRATION DONE"

# echo " ---- STARTING SERVERS ----"

uvicorn main:app --host 0.0.0.0 --port 8000

echo " --- SERVER STARTED ----"
