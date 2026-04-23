from opentrons import protocol_api
import urllib.request
import urllib.parse
import json
import time

metadata = {
    'protocolName': 'Dispose pipette tip',
    'author': 'me',
    'description': '1000ul',
    'apiLevel': '2.13'
}

def run(protocol: protocol_api.ProtocolContext):
    server_url = "REPLACE_WITH_IP:5000/capture"

    for step in range(3):
        # Send POST request to laptop server
        data = urllib.parse.urlencode({}).encode()  # empty data
        try:
            with urllib.request.urlopen(server_url, data=data, timeout=10) as response:
                result = json.loads(response.read().decode())
            protocol.comment(f"Step {step+1}: Received {result}")
        except Exception as e:
            protocol.comment(f"Step {step+1}: Request failed: {e}")
            continue

        # Example: use result to decide pipette volume
        volume = result.get("pipette_volume", 50)
        protocol.comment(f"Pipetting volume: {volume} µL")
        time.sleep(1)