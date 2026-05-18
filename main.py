import os
# Essential flags for Linux VM rendering
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["OPEN3D_CPU_RENDERING"] = "true"

import open3d.visualization.gui as gui
from ui_window import RegistrationApp

def main() -> None:
    gui.Application.instance.initialize()
    app = RegistrationApp()
    gui.Application.instance.run()

if __name__ == "__main__":
    main()