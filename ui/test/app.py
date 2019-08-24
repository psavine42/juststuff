import toga
# from toga import Canvas, Box
# from toga.style import Pack
import toga_gtk.widgets
import gi
from gi.repository import Gtk
"""
conda install -c anaconda pango
conda install -c ska pygtk
"""
def button_handler(widget):
    print("hello")


def build(app):
    box = toga.Box()

    button = toga.Button('Hello world', on_press=button_handler)
    button.style.padding = 50
    button.style.flex = 1
    box.add(button)

    return box


class StartApp(toga.App):
    def startup(self):
        # Main window of the application with title and size
        self.main_window = toga.MainWindow(title=self.name, size=(148, 250))

        # Create canvas and draw tiberius on it
        self.canvas = toga.Canvas(style=toga.style.Pack(flex=1))
        box = toga.Box(children=[self.canvas])
        # self.draw_tiberius()

        # Add the content on the main window
        self.main_window.content = box

        # Show the main window
        self.main_window.show()


def main():
    return toga.App('First App',
                    'org.beeware.helloworld', startup=build)

def main2():
    return StartApp('Tutorial 4', 'org.beeware.helloworld')

if __name__ == '__main__':
    main2().main_loop()

