import serial, json

ser = serial.Serial('COM19', 9600)

buffer = []
prev = None

def filter_signal(raw):
    buffer.append(raw)
    if len(buffer) > 10:
        buffer.pop(0)
    return sum(buffer) / len(buffer)

def soft_sensor(raw):
    return raw * 0.5 + 200 

def anomaly(raw):
    global prev
    if raw < 50 or raw > 4000:
        return True
    if prev and abs(raw - prev) > 500:
        return True
    prev = raw
    return False

def mint_credit(co2):
    print("✔ Credit Minted:", co2)

while True:
    line = ser.readline().decode().strip()
    
    try:
        data = json.loads(line)
        raw = data['mq135']
        
        filtered = filter_signal(raw)
        co2 = soft_sensor(filtered)
        
        if anomaly(raw):
            print("Anomaly Detected")
            continue
        
        print("CO2:", co2)

        if 300 < co2 < 2000:
            mint_credit(co2)
            
    except:
        pass
