# RTNeural UGen
A SuperCollider UGen which implements the RTNeural inference engine. This engine is designed for real-time safe inference of tensorflow and pytorch trained neural networks.

### Installation:

If you can, you probably want to use one of the pre-build binaries available on GitHub. On mac you will also need to clear the attributes so the OS doesn't think the file is trash:

```
xattr -cr <path to the plugin directory>
```

###Building:

1. Download this repository (plus the RTNeural and libsamplerate submodules) to their own directory.

```
git clone https://github.com/spluta/RTNeural_Plugin.git --recursive
```
RTNeural will load into the RTNeuralCPP directory
libsamplerate will load into the libsamplerate directory

## Building the SC Plugin On Mac (or Linux)

2. Build libsamplerate in release mode
(setting BUILD_TESTING to FALSE disables testing that makes it look like it didn't build correctly when it did)

From inside the libsamplerate sub-directory:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=FALSE ..
make
```

for a mac universal build you have to build the library universal as well. change the 3rd line to the following:
```
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=FALSE -DCMAKE_OSX_ARCHITECTURES='arm64;x86_64' ..
```

This may throw an error, but as long as the libsamplerate.a file is in the build/src directory, it has been built.


3. Build the SuperCollider Plugin:

To build the SC Plugin, you will need the SC source code downloaded to your computer. Get this from supercollider.github.io

from inside the RTNeural_SC subdirectory:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DSC_PATH=<PATH TO SuperCollider SOURCE directory> ..
cmake --build . --config Release
```

for a mac universal build:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DSC_PATH=<PATH TO SC SOURCE> -DCMAKE_OSX_ARCHITECTURES='arm64;x86_64' ..
cmake --build . --config Release
```

It should build RTNeural plugin and leave the RTNeuralUGen.scx file in the build directory

After building make sure the RTNeuralUGen.scx and the RTNeural_SC/RTNeural directory (with sc, schelp, and python files) are in the SC path, recompile the SC libary, and they should work. 


## On PC

2. Replace the libsamplerate directory with the most recent Win64 release from (https://github.com/libsndfile/libsamplerate/releases/)

 - Rename the folder "libsamplerate"

3. Build the Plugin (from the RTNeural_SC main directory):
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DSC_PATH=<PATH TO SC SOURCE> ..
cmake --build . --config Release
```

## Building the PD Plugin On Mac (or Linux)

Just like above, the libsamplerate project needs to be built before the pd plugin can be built

From the rtneural~_pd directory.

```
cmake . -B build -DPD_PATH=<PATH TO PD SOURCE>
cmake --build build
```

if you remove the -DPD_PATH variable , CMAKE will search for the pd source in the standard locations

the rtneural and rtneural~ objects will be placed in the build directory. move these to a place in your pd path.

