# RTNeural UGen
A SuperCollider UGen which implements the RTNeural inference engine. This engine is designed for real-time safe inference of tensorflow and pytorch trained neural networks.

### Installation:

If you can, you probably want to use one of the pre-build binaries available on GitHub. On mac you will also need to clear the attributes so the OS doesn't think the file is trash:

```
xattr -cr <path to the plugin directory>
```

###Building:

1. Download this repository to its own directory.

2. Download the RTNeural and libsamplerate submodules:
```
git submodule update --init --recursive
```
RTNeural will load into the RTNeuralCPP directory

## On Mac

3. Build libsamplerate in release mode (from the libsamplerate submodule directory):
(setting BUILD_TESTING to FALSE disables testing that makes it look like it didn't build correctly when it did)
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=FALSE ..
make
```

for a mac universal build you have to build the library universal as well:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=FALSE -DCMAKE_OSX_ARCHITECTURES='arm64;x86_64' ..
make
```

This may throw an error, but as long as the libsamplerate.a file is in the build/src directory, it has built.


4. Build the Plugin (from the RTNeural main directory):
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DSC_PATH=<PATH TO SC SOURCE> ..
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

After building make sure this directory the scx, sc, and schelp files are in the SC path, recompile the SC libary, and they should work. 

## On PC

3a. Replace the libsamplerate directory with the most recent Win64 release from (https://github.com/libsndfile/libsamplerate/releases/)

3b. Rename the folder "libsamplerate"

4. Build the Plugin (from the RTNeural main directory):
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DSC_PATH=<PATH TO SC SOURCE> ..
cmake --build . --config Release
```


