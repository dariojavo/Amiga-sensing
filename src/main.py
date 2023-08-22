# Copyright (c) farm-ng, inc. Amiga Development Kit License, Version 0.1
import argparse
import asyncio
import os
from typing import List
import sys
from amiga_sensing.GPS import GPS, find_gps_device
import os

# import internal libs

# Must come before kivy imports
os.environ["KIVY_NO_ARGS"] = "1"

# gui configs must go before any other kivy import
from kivy.config import Config  # noreorder # noqa: E402

Config.set("graphics", "resizable", False)
Config.set("graphics", "width", "1280")
Config.set("graphics", "height", "800")
Config.set("graphics", "fullscreen", "false")
Config.set("input", "mouse", "mouse,disable_on_activity")
Config.set("kivy", "keyboard_mode", "systemanddock")

# kivy imports
from kivy.app import App  # noqa: E402
from kivy.lang.builder import Builder  # noqa: E402
import numpy as np
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy_garden.mapview import MapView, MapMarker
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
import csv
import time
from farm_ng.oak import oak_pb2
from farm_ng.oak.camera_client import OakCameraClient
from farm_ng.service import service_pb2
from farm_ng.service.service_client import ClientConfig
from turbojpeg import TurboJPEG
import grpc
import cv2
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import threading
import json
import subprocess
from kivy.properties import NumericProperty
from kivy.uix.screenmanager import Screen
import struct

def get_timestamp_with_milliseconds():
    timestamp = time.time()
    milliseconds = int((timestamp - int(timestamp)) * 1000)
    return timestamp, milliseconds


class TemplateApp(App):
    """Base class for the main Kivy app."""

    def __init__(self, path: str, address: str, port1: int, port2: int, auto_exposure: bool, exposure_time: int, iso: int, stream_every_n: int, simulation: bool) -> None:
        super().__init__()
        self.t = self.t1 = 0 # erase
        self.counter: int = 0
        self.marker = None  # Initialize the marker attribute
        self.longitude = -122.4194
        self.latitude = 37.7749
        self.start_counter = False
        self.path = path
        self.address = address
        self.port1= port1
        self.port2 = port2
        self.auto_exposure = auto_exposure
        self.exposure_time = exposure_time
        self.iso = iso
        self.stream_every_n = stream_every_n
        gps_device = find_gps_device()
        if gps_device:
            print(f"GPS device found: {gps_device}")
        else:
            print("No GPS device found.")
            gps_device = None
        self.simulation = simulation
        self.gps = GPS(gps_device, simulation=self.simulation)
        self.image_decoder = TurboJPEG()
        self.tasks: List[asyncio.Task] = []
        self.csv_filename = 'Amiga'  
        # self.stop_threads = threading.Event()
        self.camera_parameters = False
        self.camera_parameters2 = False

    def on_exit_btn(self):
        """Stops the running kivy application and cancels all running tasks."""
        # Cancel all running tasks
        self.stop_threads.set() #Signal the threads to stop
        self.gps.stop() #Signal to stop gps thread
        self.image_queue.put(None)
        self.gps_queue.put(None)
        self.gps_writer_thread.join()
        self.image_writer_thread.join()

        App.get_running_app().stop()

    def btn_clk(self):
        self.start_counter = not self.start_counter  # Toggle the counter state
        if self.start_counter:
            # # Generate a timestamp-based filename for the CSV
            timestamp = int(time.time())
            self.new_path = self.path + f'/Amiga_record_{timestamp}'
            os.makedirs(self.new_path, exist_ok= True)

            self.csv_filename = self.new_path + f'/Amiga_record_{timestamp}.csv'

            # Header for the CSV file
            header = ["Timestamp", "Camera ID", "Image Name", "GPS file", "Latitude", "Longitude"]

            # Open the CSV file for writing (in append mode 'a')
            with open(self.csv_filename, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)

                # Write the header row to the CSV file
                csv_writer.writerow(header)
            
            self.root.ids.record_button.text = 'Stop'
            self.root.ids.record_button.color = [1, 0, 0, 1]  # Change the button's color to red
            # Create a folder for saving images specific to the camera ID
            image_save_path = self.new_path + '/oak0/'
            os.makedirs(image_save_path, exist_ok=True)

            
            image_save_path =  self.new_path + '/oak1/'
            os.makedirs(image_save_path, exist_ok=True)

            gps_save_path = self.new_path + '/gps/'
            os.makedirs(gps_save_path, exist_ok=True)           

        else:
            self.csv_filename = 'Amiga'  
            self.root.ids.record_button.text = 'Record'
            self.root.ids.record_button.color = [0, 1, 1, .67]  # Change the button's color back to original


    def build(self):
        root =  Builder.load_file("res/main.kv")
        
        self.mapview = root.ids.map_view
        
        self.oak1 = root.ids.oak1
        self.oak1.allow_stretch = True
        self.oak1.keep_ratio = False

        self.oak0 = root.ids.oak0
        self.oak0.allow_stretch = True
        self.oak0.keep_ratio = False

        self.oak1_d = root.ids.oak1_disparity
        self.oak1_d.allow_stretch = True
        self.oak1_d.keep_ratio = False

        self.oak0_d = root.ids.oak0_disparity
        self.oak0_d.allow_stretch = True
        self.oak0_d.keep_ratio = False

        #For the camera parameters screen

        self.oak1_2 = root.ids.oak1_2
        self.oak1_2.allow_stretch = True
        self.oak1_2.keep_ratio = False

        self.oak0_2 = root.ids.oak0_2
        self.oak0_2.allow_stretch = True
        self.oak0_2.keep_ratio = False

        self.oak1_d_2 = root.ids.oak1_disparity2
        self.oak1_d_2.allow_stretch = True
        self.oak1_d_2.keep_ratio = False

        self.oak0_d_2 = root.ids.oak0_disparity2
        self.oak0_d_2.allow_stretch = True
        self.oak0_d_2.keep_ratio = False

        self.dropdown = None

        return root
    
    def start_threads(self):
        self.stop_threads = threading.Event() 
        self.gps_queue = Queue()
        self.gps_writer_thread = threading.Thread(target=self.write_to_csv, args=(self.csv_filename, self.gps_queue,))
        self.gps_writer_thread.start()
        # Create a queue to hold the images and CSV rows
        self.image_queue = Queue()
        self.image_writer_thread = None
        # At the start of your program, start the thread
        self.image_writer_thread = threading.Thread(target=self.write_image_and_csv, args=(self.csv_filename, self.image_queue,))
        self.image_writer_thread.start()
        
        
    # Define a function that will handle writing to the CSV and saving images

    def write_image_and_csv(self, filename, queue):
        try:
            while not self.stop_threads.is_set():
                print("Queue size:", queue.qsize())
                item = queue.get()
                if item is None:
                    break
                timestamp, camera_id, img, image_path, row = item
                # cv2.imwrite(image_path, img)
                height, width, channels = img.shape
                dtype_str = str(img.dtype)
                with open(image_path, 'wb') as f:
                    # Write image properties as metadata
                    f.write(struct.pack('iiii', width, height, channels, len(dtype_str)))
                    f.write(dtype_str.encode())
                    # Write image data
                    f.write(img.tobytes())
                if filename == 'Amiga':
                    filename = self.csv_filename
                with open(filename, 'a', newline='') as csvfile_image:
                    csv_writer = csv.writer(csvfile_image)
                    csv_writer.writerow(row)
                queue.task_done()
        except Exception as e:
            print(f"Exception in write_to_csv: {e}")
    def write_to_csv(self, csv_filename, gps_queue):
        try:
            while not self.stop_threads.is_set():
                item = gps_queue.get()
                if item is None:
                    break
                gps_file_name, row = item
                # Writing to sample.json
                gps_sample = {
                                "year":self.geo.year,
                                "month":self.geo.month,
                                "day":self.geo.day,
                                "hour":self.geo.hour,
                                "sec":self.geo.sec,
                                "nano":self.geo.nano,
                                "lon":self.geo.lon,
                                "lat":self.geo.lat,
                                "height": self.geo.height,
                                "headMot":self.geo.headMot,
                }
                with open(self.new_path + gps_file_name, "w") as outfile:
                    json.dump(gps_sample, outfile)
                if csv_filename == 'Amiga':
                    csv_filename = self.csv_filename           
                with open(csv_filename, 'a', newline='') as csvfile_image:
                    csv_writer = csv.writer(csvfile_image)
                    csv_writer.writerow(row)
                gps_queue.task_done()
        except Exception as e:
            print(f"Exception in write_to_csv: {e}")

    def get_usb_devices(self):
        # This method will list USB devices. Modify according to your OS.
        try:
            result = subprocess.check_output(["lsusb"]).decode("utf-8")
            devices = [line for line in result.split("\n") if line]
            return devices
        except:
            # In case of error or if the OS doesn't support the 'lsusb' command, return an empty list
            return []

    def open_usb_dropdown(self, instance):
        if not self.dropdown:
            self.dropdown = DropDown()

            usb_devices = self.get_usb_devices()
            for device in usb_devices:
                btn = Button(text=device, size_hint_y=None, height=44, font_size='10sp')
                btn.bind(on_release=lambda btn: self.set_dropdown_value_and_close(btn.text))
                self.dropdown.add_widget(btn)

        # Opening the dropdown
        self.dropdown.open(instance)

    def set_dropdown_value_and_close(self, value):
        self.root.ids.usb_dropdown_trigger.text = value
        self.dropdown.dismiss()

    async def app_func(self):
        async def run_wrapper() -> None:
            # we don't actually need to set asyncio as the lib because it is
            # the default, but it doesn't hurt to be explicit
            await self.async_run(async_lib="asyncio")
            for task in self.tasks:
                task.cancel()

        self.start_threads()  # start the threads here
        # configure the camera client
        config = ClientConfig(address=self.address, port=self.port1)
        client = OakCameraClient(config)

        # configure the camera client
        config2 = ClientConfig(address=self.address, port=self.port2)
        client2 = OakCameraClient(config2)
        
        #Start GPS
        self.gps.start()

        # Stream camera frames
        self.tasks.append(asyncio.ensure_future(self.stream_camera(client, 50051)))
        self.tasks.append(asyncio.ensure_future(self.stream_camera(client2, 50052)))
        self.tasks.append(asyncio.ensure_future(self.update_gps_position()))
        return await asyncio.gather(run_wrapper(), *self.tasks) 


    async def stream_camera(self, client: OakCameraClient, port) -> None:
        """This task listens to the camera client's stream and populates the tabbed panel with all 4 image streams
        from the oak camera."""
        while self.root is None:
            await asyncio.sleep(0.01)

        response_stream = None

        while True:
            print(self.image_writer_thread.is_alive())  # will print True if thread is running, False if it's stopped
            # check the state of the service
            state = await client.get_state()
            
            # Update_camera_values = {
            # 'auto_exposure': self.root.ids.auto_exposure.active,
            # 'exposure_time': self.root.ids.exposure_time.value,
            # 'iso_value': self.root.ids.iso.value,
            # 'lens_pos' : self.root.ids.lens.value,
            # 'fps' : self.root.ids.fps.value
            # }
            
            # print(Update_camera_values)
                
            if state.value not in [
                service_pb2.ServiceState.IDLE,
                service_pb2.ServiceState.RUNNING,
            ]:
                # Cancel existing stream, if it exists
                if response_stream is not None:
                    response_stream.cancel()
                    response_stream = None
                print("Camera service is not streaming or ready to stream")
                await asyncio.sleep(0.1)
                continue

            # Create the stream
            if response_stream is None:
                # # Update camera parameters
                # Update_camera_values = {
                # 'auto_exposure': self.root.ids.auto_exposure.active,
                # 'exposure_time': self.root.ids.exposure_time.value,
                # 'iso_value': self.root.ids.iso.value,
                # 'lens_pos' : self.root.ids.lens.value,
                # 'fps' : self.root.ids.fps.value
                # }
                
                # print(Update_camera_values)
                
                # # Create a new instance of CameraSettings with desired parameters
                # new_rgb_settings = new_mono_settings = oak_pb2.CameraSettings(
                #     auto_exposure = Update_camera_values['auto_exposure'],         # Set auto exposure
                #     exposure_time = int(Update_camera_values['exposure_time']),         # Assume this represents 1000ms or 1 second. Adjust based on your needs.
                #     iso_value = int(Update_camera_values['iso_value']),              # ISO value
                #     lens_pos = int(Update_camera_values['lens_pos'])
                # )
                print("Exposure", self.auto_exposure)
                # Create a new instance of CameraSettings with desired parameters
                new_rgb_settings = new_mono_settings = oak_pb2.CameraSettings(
                    auto_exposure = self.auto_exposure,         # Set auto exposure
                    exposure_time = self.exposure_time,         # Assume this represents 1000ms or 1 second. Adjust based on your needs.
                    iso_value = self.iso              # ISO value
                )
                               
                # Assuming new_rgb_settings is a protobuf object of CameraSettings type
                client.update_rgb_settings(new_rgb_settings)

                # Similarly for mono camera
                client.update_mono_settings(new_mono_settings)
                await asyncio.sleep(1)
                # Send modified settings to the camera
                response = await client.send_settings()

                if response:
                    print(f"Ok, parameters have been updated for {port}")

                response_stream = client.stream_frames(every_n=self.stream_every_n)

            try:
                # try/except so app doesn't crash on killed service
                response: oak_pb2.StreamFramesReply = await response_stream.read()
                assert response and response != grpc.aio.EOF, "End of stream"
            except Exception as e:
                print(e)
                response_stream.cancel()
                response_stream = None
                continue

            # get the sync frame
            frame: oak_pb2.OakSyncFrame = response.frame

            # data: bytes = getattr(frame, "rgb").image_data
            for view_name in ["rgb", "disparity"]:
                try:
                    # Decode the image and render it in the correct kivy texture
                    img = self.image_decoder.decode(
                        getattr(frame, view_name).image_data
                    )
                    texture = Texture.create(
                        size=(img.shape[1], img.shape[0]), icolorfmt="bgr"
                    )
                    texture.flip_vertical()
                    texture.blit_buffer(
                        img.tobytes(),
                        colorfmt="bgr",
                        bufferfmt="ubyte",
                        mipmap_generation=False,
                    )
                    if port == 50051:
                        if view_name == 'rgb':
                            self.oak0.texture = texture
                            self.oak0_2.texture = texture
                            elapsed = time.time() - self.t
                            self.t = time.time()
                            print('Camera Oak0 Hz:', 1/elapsed)
                        else:
                            self.oak0_d.texture = texture
                            self.oak0_d_2.texture = texture
                        camera_id = 'oak0'
                    elif port == 50052:
                        if view_name == 'rgb':
                            self.oak1.texture = texture
                            self.oak1_2.texture = texture
                            elapsed1 = time.time() - self.t1
                            self.t1 = time.time()
                            print('Camera Oak1 Hz:', 1/elapsed1)
                        else:
                            self.oak1_d.texture = texture
                            self.oak1_d_2.texture = texture
                        camera_id = 'oak1'

                    if self.start_counter:
                            timestamp, milliseconds = get_timestamp_with_milliseconds()
                            image_name =  f'/{camera_id}/{view_name}_image_{timestamp}.txt'
                            image_path = self.new_path + image_name
                            gps_file_name = 'None'#f'image_{camera_id}_{int(timestamp)}_{milliseconds:03d}.jpg'
                            latitude = 'None'
                            longitude = 'None'
                            row = [timestamp, camera_id, image_name, gps_file_name, latitude, longitude]
                            self.image_queue.put((timestamp, camera_id, img, image_path, row))
                        # If you want to stop the thread, for example when self.start_counter is False
                    else: 
                            self.image_queue.put(None)
                            self.image_writer_thread.join()
                            # Once the thread is joined, create a new thread for the next possible round
                            self.image_writer_thread = threading.Thread(target=self.write_image_and_csv, args=(self.csv_filename, self.image_queue,))
                            self.image_writer_thread.start()

                            
                except Exception as e:
                    print(e)

    async def update_gps_position(self):
        while self.root is None:
            await asyncio.sleep(0.01)

        while True:

            self.geo = self.gps.get_gps_data()
            if self.geo is not None:
                latitude = self.geo.lat
                longitude = self.geo.lon
                timestamp, milliseconds = get_timestamp_with_milliseconds()
                gps_file_name =  f'/gps/gps_data_{timestamp}.json' 
                
                if self.start_counter:

                # Prepare data to write
                    gps_data = [timestamp, 'None', 'None', gps_file_name, latitude, longitude]
                    self.gps_queue.put((gps_file_name, gps_data))                    

                else:
                    self.gps_queue.put(None)
                    self.gps_writer_thread.join()
                    # Once the thread is joined, create a new thread for the next possible round
                    self.gps_writer_thread = threading.Thread(target=self.write_to_csv, args=(self.csv_filename, self.gps_queue,))
                    self.gps_writer_thread.start()

                # Update the marker position on the map
                if self.marker is not None:
                    self.mapview.remove_marker(self.marker)  # Remove previous marker
                self.marker = MapMarker(lat=latitude, lon=longitude)
                self.mapview.add_marker(self.marker)

                # Center the map view on the current GPS position
                self.mapview.center_on(latitude, longitude)
                self.mapview.zoom = 15
            await asyncio.sleep(0.1)
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="template-app")
    parser.add_argument("--port1", type=int, default=50051, required=False, help="The camera1 port.")
    parser.add_argument("--port2", type=int, default=50052, required=False, help="The camera2 port.")
    parser.add_argument("--auto_exposure", type=bool, default=False, required=False, help="Auto exposure")
    parser.add_argument("--exposure_time", type=int, default=10000, required=False, help="Exposure time")
    parser.add_argument("--iso", type=int, default=100, required=False, help="ISO gain")
    parser.add_argument("--address", type=str, default="localhost", help="The camera address")
    parser.add_argument("--stream-every-n", type=int, default=1, help="Streaming frequency")
    parser.add_argument("--simulation", type=bool, default=False, help="Run simulation mode")
    # Add additional command line arguments here
    parser.add_argument("--path", type=str, default='.', required=False, help="The camera port.")
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(TemplateApp(args.path, args.address, args.port1, args.port2, args.auto_exposure, args.exposure_time, args.iso, args.stream_every_n, args.simulation).app_func())
    except asyncio.CancelledError:
        pass
    loop.close()
