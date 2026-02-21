import serial
import json
import csv

ser = serial.Serial('COM19', 9600)

with open('data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['mq135', 'co2'])

    for i in range(100):
        line = ser.readline().decode().strip()

        try:
            data = json.loads(line)
            raw = data['mq135']

            print("Raw:", raw)
            co2 = 400

            writer.writerow([raw, co2])

        except:
            pass
