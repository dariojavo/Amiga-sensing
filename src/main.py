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


# class in which we are defining action on click
class RootWidget(BoxLayout):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app  # store the app instance

    def btn_clk(self):
        self.app.start_counter = True  # start the counter when the button is clicked
        # self.lbl.text = "You have been pressed"

class TemplateApp(App):
    """Base class for the main Kivy app."""

    def __init__(self) -> None:
        super().__init__()

        self.counter: int = 0
        self.marker = None  # Initialize the marker attribute
        self.async_tasks: List[asyncio.Task] = []
        self.longitude = -122.4194
        self.latitude = 37.7749
        self.start_counter = False

    def build(self):
        root =  Builder.load_file("res/main.kv")
        # Right half with a map view
        root = RootWidget(self)
        self.mapview = root.ids.map_view

        self.image = root.ids.image

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

        # Placeholder task
        self.async_tasks.append(asyncio.ensure_future(self.template_function()))

        return await asyncio.gather(run_wrapper(), *self.async_tasks)

    async def template_function(self) -> None:
        """Placeholder forever loop."""
        while self.root is None:
            await asyncio.sleep(0.01)

        while True:
            await asyncio.sleep(1.0)

            if self.start_counter:
                # increment the counter using internal libs and update the gui
                self.counter = ops.add(self.counter, 1)
                self.root.ids.counter_label.text = (
                    f"{'Tic' if self.counter % 2 == 0 else 'Tac'}: {self.counter}"
                )

                # Update the noisy image and map marker
                self.update_noisy_image()
                self.update_gps_position()
                

    def update_noisy_image(self):
        # Generate a random noisy image (300x300 with random values between 0 and 255)
        noise_img = np.random.randint(0, 256, (1920, 1080, 3), dtype=np.uint8)

        # Display the noisy image in the left half
        self.display_image(noise_img)

    def display_image(self, img_array):
        # Convert NumPy array to Kivy Texture
        h, w, _ = img_array.shape
        texture = Texture.create(size=(w, h), colorfmt='rgb')

        # Flip the image vertically (Kivy's image origin is bottom-left)
        img_array = np.flipud(img_array)

        # Copy the image data to the texture
        texture.blit_buffer(img_array.tobytes(), colorfmt='rgb', bufferfmt='ubyte')

        # Set the texture for the Image widget
        self.image.texture = texture

    def update_gps_position(self):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="template-app")

    # Add additional command line arguments here

    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(TemplateApp().app_func())
    except asyncio.CancelledError:
        pass
    loop.close()
