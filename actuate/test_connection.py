"""
Quick Test Script - Verify Arduino Connection
Run this before the full system to test hardware

Usage:
    python test_connection.py
"""

import serial
import serial.tools.list_ports
import time


def find_ports():
    """List all available serial ports"""
    print("\n" + "="*60)
    print("SCANNING FOR SERIAL PORTS")
    print("="*60)
    
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("❌ No serial ports found!")
        print("\nTroubleshooting:")
        print("  1. Is Arduino connected via USB?")
        print("  2. Are Arduino drivers installed?")
        print("  3. Try a different USB cable/port")
        return None
    
    print(f"\n✅ Found {len(ports)} port(s):\n")
    
    for i, port in enumerate(ports, 1):
        print(f"  {i}. {port.device}")
        print(f"     Description: {port.description}")
        print(f"     Hardware ID: {port.hwid}")
        print()
    
    return ports


def test_arduino_connection(port):
    """Test connection to Arduino"""
    print("="*60)
    print(f"TESTING CONNECTION: {port}")
    print("="*60)
    
    try:
        print("\n1. Opening serial port...")
        ser = serial.Serial(port, 9600, timeout=2)
        print("   ✅ Port opened")
        
        print("\n2. Waiting for Arduino to initialize...")
        time.sleep(2)
        print("   ✅ Ready")
        
        print("\n3. Clearing input buffer...")
        ser.reset_input_buffer()
        print("   ✅ Buffer cleared")
        
        print("\n4. Listening for data (10 seconds)...")
        data_received = False
        valid_readings = 0
        start_time = time.time()
        
        while (time.time() - start_time) < 10:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                if line:
                    print(f"   📡 Received: {line}")
                    
                    # Check if it's sensor data (6 comma-separated values)
                    parts = line.split(',')
                    if len(parts) == 6:
                        try:
                            temp = float(parts[0])
                            co2 = float(parts[1])
                            humidity = float(parts[2])
                            flow = float(parts[3])
                            weight = float(parts[4])
                            fan = int(parts[5])
                            
                            print(f"   ✅ Valid data:")
                            print(f"      Temperature: {temp}°C")
                            print(f"      CO2: {co2} ppm")
                            print(f"      Humidity: {humidity}%")
                            print(f"      Flow: {flow} L/min")
                            print(f"      Weight: {weight} g")
                            print(f"      Fan RPM: {fan}")
                            
                            valid_readings += 1
                            data_received = True
                        except ValueError:
                            print(f"   ⚠️ Invalid format")
            
            time.sleep(0.1)
        
        if valid_readings > 0:
            print(f"\n✅ SUCCESS! Received {valid_readings} valid sensor readings")
        elif data_received:
            print(f"\n⚠️ Data received but format incorrect")
        else:
            print(f"\n❌ No data received")
            print("\nTroubleshooting:")
            print("  1. Is the Arduino sketch uploaded?")
            print("  2. Check Serial Monitor at 9600 baud")
            print("  3. Restart Arduino and try again")
        
        # Test fan control
        if valid_readings > 0:
            print("\n5. Testing fan control...")
            
            print("   Sending: Fan ON (1)")
            ser.write(b'1')
            ser.flush()
            time.sleep(2)
            
            print("   Sending: Fan OFF (0)")
            ser.write(b'0')
            ser.flush()
            time.sleep(1)
            
            print("   ✅ Commands sent (check if fan responded)")
        
        ser.close()
        print("\n✅ Test complete!")
        return valid_readings > 0
        
    except serial.SerialException as e:
        print(f"\n❌ Serial error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def main():
    """Main test routine"""
    print("\n" + "🔧"*30)
    print("ARDUINO CONNECTION TEST")
    print("🔧"*30)
    
    print("\nThis script will:")
    print("  1. Find available serial ports")
    print("  2. Connect to Arduino")
    print("  3. Verify sensor data")
    print("  4. Test fan commands")
    
    # Find ports
    ports = find_ports()
    
    if not ports:
        return
    
    # Select port
    if len(ports) == 1:
        selected = ports[0].device
        print(f"\nAuto-selecting only port: {selected}")
    else:
        choice = input(f"\nSelect port (1-{len(ports)}): ").strip()
        try:
            selected = ports[int(choice) - 1].device
        except (ValueError, IndexError):
            print("❌ Invalid selection")
            return
    
    # Test connection
    success = test_arduino_connection(selected)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if success:
        print("\n✅ PASS - Arduino is working correctly!")
        print("\nNext steps:")
        print("  1. Run: python carbon_capture_brain.py")
        print("  2. Follow the prompts")
        print("  3. System will start collecting data")
    else:
        print("\n❌ FAIL - Issues detected")
        print("\nTroubleshooting steps:")
        print("  1. Upload carbon_capture_arduino.ino")
        print("  2. Check wiring (DS18B20 + MQ135)")
        print("  3. Open Arduino Serial Monitor (9600 baud)")
        print("  4. Run this test again")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()