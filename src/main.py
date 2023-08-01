# Copyright (c) farm-ng, inc. Amiga Development Kit License, Version 0.1
import argparse
import asyncio
import os
from typing import List

from amiga_sensing import ops

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
import csv
import time
from farm_ng.oak import oak_pb2
from farm_ng.oak.camera_client import OakCameraClient
from farm_ng.service import service_pb2
from farm_ng.service.service_client import ClientConfig
from turbojpeg import TurboJPEG
import grpc

# class in which we are defining action on click
class RootWidget(BoxLayout):
    def __init__(self, app, path, **kwargs):
        super().__init__(**kwargs)
        self.app = app  # store the app instance
        self.path = path

    def btn_clk(self):
        self.app.start_counter = not self.app.start_counter  # Toggle the counter state
        if self.app.start_counter:
            self.ids.record_button.text = 'Stop'
            self.ids.record_button.color = [1, 0, 0, 1]  # Change the button's color to red
            # Create new csv file for data recording
            
            # Generate a timestamp-based filename for the CSV
            timestamp = int(time.time())
            new_path = self.path + f'/Amiga_record_{timestamp}'
            os.makedirs(new_path, exist_ok= True)
            # Create a folder for saving images specific to the camera ID
            image_save_path = new_path + '/camera_OAK0/'
            os.makedirs(image_save_path, exist_ok=True)

            image_save_path =  new_path + '/camera_OAK1/'
            os.makedirs(image_save_path, exist_ok=True)

            gps_save_path = new_path + '/gps-sparkfun/'
            os.makedirs(gps_save_path, exist_ok=True)           

            csv_filename = new_path + f'/Amiga_record_{timestamp}.csv'
    
            # Header for the CSV file
            header = ["Timestamp", "Camera ID", "Image Name", "GPS file", "Latitude", "Longitude"]

            # Open the CSV file for writing (in append mode 'a')
            with open(csv_filename, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)

                # Write the header row to the CSV file
                csv_writer.writerow(header)

        else:
            self.ids.record_button.text = 'Record'
            self.ids.record_button.color = [0, 1, 1, .67]  # Change the button's color back to original
class TemplateApp(App):
    """Base class for the main Kivy app."""

    def __init__(self, path: str, address: str, port: int, stream_every_n: int) -> None:
        super().__init__()

        self.counter: int = 0
        self.marker = None  # Initialize the marker attribute
        self.async_tasks: List[asyncio.Task] = []
        self.longitude = -122.4194
        self.latitude = 37.7749
        self.start_counter = False
        self.path = path
        self.address = address
        self.port = port
        self.stream_every_n = stream_every_n

        self.image_decoder = TurboJPEG()
        self.tasks: List[asyncio.Task] = []

    def build(self):
        root =  Builder.load_file("res/main.kv")
        # Right half with a map view
        root = RootWidget(self, path=self.path)
        self.mapview = root.ids.map_view

        self.image = root.ids.image
        self.rgb = root.ids.rgb
        self.rgb.allow_stretch = True
        self.rgb.keep_ratio = False

        # self.image.size_hint = (0.5, 1)
        self.image.allow_stretch = True
        self.image.keep_ratio = False
        
        return root


    def on_exit_btn(self) -> None:
        """Kills the running kivy application."""
        App.get_running_app().stop()

    async def app_func(self):
        async def run_wrapper() -> None:
            # we don't actually need to set asyncio as the lib because it is
            # the default, but it doesn't hurt to be explicit
            await self.async_run(async_lib="asyncio")
            for task in self.async_tasks:
                task.cancel()

        # # Placeholder task
        # self.async_tasks.append(asyncio.ensure_future(self.template_function()))

        # return await asyncio.gather(run_wrapper(), *self.async_tasks)

        # configure the camera client
        config = ClientConfig(address=self.address, port=self.port)
        client = OakCameraClient(config)

        # configure the camera client
        config2 = ClientConfig(address=self.address, port=50052)
        client2 = OakCameraClient(config2)

        # Stream camera frames
        self.tasks.append(asyncio.ensure_future(self.stream_camera(client, 50051)))
        self.tasks.append(asyncio.ensure_future(self.stream_camera(client2, 50052)))
        self.tasks.append(asyncio.ensure_future(self.update_gps_position()))
        return await asyncio.gather(run_wrapper(), *self.tasks) 

    # async def template_function(self) -> None:
    #     """Placeholder forever loop."""
    #     while self.root is None:
    #         await asyncio.sleep(0.01)

    #     while True:
    #         await asyncio.sleep(1.0)
    #         async def run_wrapper():
    #             # we don't actually need to set asyncio as the lib because it is
    #             # the default, but it doesn't hurt to be explicit
    #             await self.async_run(async_lib="asyncio")
    #             for task in self.tasks:
    #                 task.cancel()

    #         # await asyncio.sleep(1.0)

    #         if self.start_counter:
    #             # # increment the counter using internal libs and update the gui
    #             # self.counter = ops.add(self.counter, 1)
    #             # self.root.ids.counter_label.text = (
    #             #     f"{'Tic' if self.counter % 2 == 0 else 'Tac'}: {self.counter}"
    #             # )

    #             # Update the noisy image and map marker
    #             # self.update_noisy_image()
    #             self.update_gps_position()

    #             # configure the camera client
    #             config = ClientConfig(address=self.address, port=self.port)
    #             client = OakCameraClient(config)

    #             # Stream camera frames
    #             self.tasks.append(asyncio.ensure_future(self.stream_camera(client)))

    #             return await asyncio.gather(run_wrapper(), *self.tasks)             

    async def stream_camera(self, client: OakCameraClient, port) -> None:
        """This task listens to the camera client's stream and populates the tabbed panel with all 4 image streams
        from the oak camera."""
        while self.root is None:
            await asyncio.sleep(0.01)

        response_stream = None

        while True:
            # check the state of the service
            state = await client.get_state()

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

            # get image and show
            for view_name in ["rgb", "disparity", "left", "right"]:
                # Skip if view_name was not included in frame
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
                        self.image.texture = texture
                    elif port == 50052:
                        self.rgb.texture = texture
                
                except Exception as e:
                    print(e)


    # def update_noisy_image(self):
    #     # Generate a random noisy image (300x300 with random values between 0 and 255)
    #     noise_img = np.random.randint(0, 256, (1920, 1080, 3), dtype=np.uint8)

    #     # Display the noisy image in the left half
    #     self.display_image(noise_img)

    # def display_image(self, img_array):
    #     # Convert NumPy array to Kivy Texture
    #     h, w, _ = img_array.shape
    #     texture = Texture.create(size=(w, h), colorfmt='rgb')

    #     # Flip the image vertically (Kivy's image origin is bottom-left)
    #     img_array = np.flipud(img_array)

    #     # Copy the image data to the texture
    #     texture.blit_buffer(img_array.tobytes(), colorfmt='rgb', bufferfmt='ubyte')

    #     # Set the texture for the Image widget
    #     self.image.texture = texture

    async def update_gps_position(self):
        while self.root is None:
            await asyncio.sleep(0.01)

        while True:
            # In this example, we'll use dummy GPS coordinates.
            # You should replace these with real GPS coordinates if available.
            self.latitude = self.latitude + 0.0001  # San Francisco, CA, USA
            self.longitude = self.longitude + 0.0001
            latitude = self.latitude
            longitude = self.longitude
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
    parser.add_argument("--port", type=int, default=50051, required=False, help="The camera port.")
    parser.add_argument(
        "--address", type=str, default="localhost", help="The camera address"
    )
    parser.add_argument(
        "--stream-every-n", type=int, default=1, help="Streaming frequency"
    )
    # Add additional command line arguments here
    parser.add_argument("--path", type=str, default='/data/data_recording/', required=False, help="The camera port.")
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(TemplateApp(args.path, args.address, args.port, args.stream_every_n).app_func())
    except asyncio.CancelledError:
        pass
    loop.close()
