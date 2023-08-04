import time
import serial
import struct
from ublox_gps import UbloxGps
from ublox_gps import sparkfun_predefines as sp
from multiprocessing import Process,Queue
import piexif
from fractions import Fraction

import datetime
import json

import glob
import subprocess
import os

assets_path = os.path.join(os.path.dirname(__file__),"../../src/assets/")

class ObjectFromDict:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_product_attribute(device_path):
    command = ['udevadm', 'info', '-q', 'all', '-a', '-n', device_path]
    output = subprocess.check_output(command, universal_newlines=True)
    lines = output.splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith('ATTRS{product}'):
            product = line.strip().split('==')[-1]
            return product
    return None

def find_gps_device(gps_product="u-blox"):
    devices = glob.glob('/dev/ttyACM*')
    for device in devices:
        
        try:
           product = get_product_attribute(device_path=device) 
           if gps_product in product:
                print(f"Found {product} on {device}")
                return device
        except IOError:
            pass

    return None

class GPS:
    def __init__(self,update_ms=300,simulation=False):
        try:
            device = find_gps_device()
            self.gps_port = serial.Serial(device, baudrate=38400, timeout=1)
            self.gps = UbloxGps(self.gps_port)
        except Exception as e:
            print(e)
            self.gps_port = None
            self.gps = None
        self.longitude = -122.4194
        self.latitude = 37.7749
        # if self.gps:
        #     # Setup samping rate
        #     ubx_id = 0x08
        #     ubx_payload = struct.pack("<HHH", update_ms, 1, 0)
        #     # Send the UBX-CFG-RATE message
        #     self.gps.send_message(sp.CFG_CLS, ubx_id, ubx_payload)

        self.simulation = simulation
        self.queue:Queue = Queue() # FIFO Queue
        self.running = False

        self.geo:object = None

        # Dry run
        if 0:
            self.update_gps()
            _ = self.get_gps_data()

    def start(self):
        self.process = Process(target=self._run, args=())
        self.running = True 
        self.process.start()
    
    def _run(self):
        if self.gps == None or self.simulation:
            print("GPS module not detected. Verify the connection.")
            print("Running GPS simulation mode")
            
            # Runing GPS Simulation

            while self.running:
                # Get the current time
                current_time = datetime.datetime.now()
                geo_dict = {
                            "year":current_time.year,
                            "month":current_time.month,
                            "day":current_time.day,
                            "hour":current_time.hour,
                            "min":current_time.min,
                            "sec":current_time.second,
                            "nano":current_time.microsecond*1000,
                            "lon":self.longitude+0.000001,
                            "lat":self.latitude + 0.000001,
                            "height": 0,
                            "headMot":0,
                } 


                try:
                    self.queue.put(ObjectFromDict(**geo_dict),timeout=0)
                except Exception as e:
                    print("GPS queue is full")
                    print(e)

        while self.running:
            self.update_gps()
            time.sleep(0.01)

    def update_gps(self):
        try:
            geo_response = self.gps.geo_coords()
        except Exception as e:
            print(e)
            geo_response = None

        if geo_response:
            if 0:
                print("UTC Time {}:{}:{}".format(geo_response.hour, geo_response.min,geo_response.sec))
                print("Longitude: ", geo_response.lon) 
                print("Latitude: ", geo_response.lat)
                print("Heading of Motion: ", geo_response.headMot)

            geo_dict = {
                        "year":geo_response.year,
                        "month":geo_response.month,
                        "day":geo_response.day,
                        "hour":geo_response.hour,
                        "min":geo_response.min,
                        "sec":geo_response.sec,
                        "nano":geo_response.nano,
                        "lon":geo_response.lon,
                        "lat":geo_response.lat,
                        "height": geo_response.height,
                        "headMot":geo_response.headMot,
            }

            try:
                self.queue.put(ObjectFromDict(**geo_dict),timeout=0)
            except Exception as e:
                print("GPS queue is full")
                print(e)

    def get_gps_data(self):
        #@TODO: Add EKF here. Aggrgate all the previous GPS data points
        
        # Get all gps points from the queue
        while not self.queue.empty():
            # print(self.queue.qsize())
            self.geo = self.queue.get()
            # print("UTC Time {}:{}:{}".format(self.geo.hour, self.geo.min,self.geo.sec))


        return self.geo
    
    def stop(self):
        self.running = False
        self.process.kill()

if __name__ == "__main__":

    # Usage
    gps_device = find_gps_device()
    if gps_device:
        print(f"GPS device found: {gps_device}")
    else:
        print("No GPS device found.")
        
    gps = GPS()
    gps.start()
    try: 
        while True:
            time.sleep(0.3)
            gps_data = gps.get_gps_data()
            if 1:
                if gps_data is not None:
                    print("UTC Time {}:{}:{}".format(gps_data.hour, gps_data.min,gps_data.sec))
                    print("Longitude: ", gps_data.lon) 
                    print("Latitude: ", gps_data.lat)
                    print("Heading of Motion: ", gps_data.headMot)
    except KeyboardInterrupt:
        gps.stop()


