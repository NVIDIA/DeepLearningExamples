#!/bin/bash

SERVER_URI=${1:-"localhost"}

echo "Waiting for TRTIS Server to be ready at http://$SERVER_URI:8000..."

live_command="curl -m 1 -L -s -o /dev/null -w %{http_code} http://$SERVER_URI:8000/api/health/live"
ready_command="curl -m 1 -L -s -o /dev/null -w %{http_code} http://$SERVER_URI:8000/api/health/ready"

current_status=$($live_command)

# First check the current status. If that passes, check the json. If either fail, loop
while [[ ${current_status} != "200" ]] || [[ $($ready_command) != "200" ]]; do

   printf "."
   sleep 1
   current_status=$($live_command)
done

echo "TRTIS Server is ready!"