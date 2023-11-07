#SplatterPy  
  
## *Procedural Gaussian Splattering using Python OpenGL bindings*  
  
This project is a Python library that uses OpenGL and GLFW to render real-time Gaussian splatters. It's designed to fill a gap in the current market, providing a simple yet effective tool for those interested in the field of real-time graphics, especially in the context of Gaussian splattering rendering, without the complexities of neural network dependencies.  
  
## Background  
  
The journey began with a fascination for real-time graphics and its ungodly horrors. A recent venture into graphics coding led to the creation of an infinite grid canvas shader that resolved significant performance issues in a Kivy application. This success sparked curiosity about what else OpenGL could offer when paired with Python bindings.  
  
The result is this project: a basic prototype graphics library built on top of OpenGL. Made to enable users to pass a Numpy array containing point coordinates and receive a real-time Gaussian splattering render. This project aims to simplify the experimentation with real-time rendering of Gaussian splatters, a feature not commonly found standalone in existing 3D software or libraries.  
  
## Requirements  
  
- Python 3.x  
- [GLFW](https://www.glfw.org/)  
- [PyOpenGL](http://pyopengl.sourceforge.net/)  
- [NumPy](https://numpy.org/)  
- [trimesh](https://trimsh.org/trimesh.html)  
  
## Installation  
  
Make sure you have Python installed and then install the required packages using pip:  
  
```
pip install glfw PyOpenGL numpy trimesh  
```  
  
## Usage  
  
Run the script directly with Python:  
```  
python main.py  
```  
This will open a GLFW window and render the generated terrain.  
  
## Features  
 - Procedural terrain generation with real-time Gaussian splattering  
 - Real-time 3D rendering using vertex, geometry, and fragment shaders  
 - Camera and perspective handling  
 - Customizable terrain features and noise parameters  
  
## Customization  
You can customize the following parameters within the script:  
  
 - width, length: Dimensions of the terrain  
 - height_variation, height_scale: Controls the elevation and scaling of the terrain  
 - octaves, persistence, lacunarity: Parameters for the noise function affecting terrain roughness  
 - fov, aspect_ratio, near_plane, far_plane: Perspective projection settings  
 - camera_pos, camera_target, camera_up: Camera position and orientation  
  
## Notes from the Creator  
This project is my first public repository, marking my foray into both graphics programming and open-source collaboration. As I learn and navigate through GitHub, I welcome suggestions, contributions, and feedback from the community. This is a learning endeavor for me, and I hope it serves as a useful tool or stepping stone for others interested in real-time graphics and rendering techniques.  
  
  
## Troubleshooting  
If you encounter any issues:  
  
 1. Ensure that all dependencies are installed correctly.  
 2. Check that your graphics drivers support the OpenGL version required.  
 3. Review the error messages in the console for any shader compilation issues.  
  
  
## Contributing  
Contributions to the project are welcome! Whether it's improving the code, refining the shaders, or enhancing the features, feel free to fork the repository, make your changes, and submit a pull request.  
  
## License  
This project is open-source and available under the MIT License. See the LICENSE file for more details.
