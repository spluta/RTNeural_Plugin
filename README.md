# RTNeural Plugin
A SuperCollider/Max/pd plugin which implements the RTNeural inference engine. This engine is designed for real-time safe inference of tensorflow and pytorch trained neural networks. It can be used to process real-time audio or to run inference at control rate. Networks can have any number of inputs and any number of outputs. Networks trained at a specific sample rate can run at any samplerate due to internal resampling using libsamplerate.

### Installation:

If you are on mac or windows, you probably want to use one of the pre-build binaries available on GitHub. The latest release is under Releases on the main github page. On mac you will also need to further purge any distrust of the builds:

SC:
```
xattr -c <path to the scx_files directory>/*.scx
```

Max
```
codesign --force --deep -s - <path to the rtneural_max/externals directory>/*.mxo
xattr -c <path to the rtneural_max/externals directory>/*.mxo
```

pd
```
xattr -c <path to the rtneural_pd/externals directory>/*.so
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

I have only been able to build this with GitHub actions. See the build-sc.yml file in the .github/workflows directory for a hope and a prayer to get this built

## Building the PD Plugin On Mac (or Linux)

Just like above, the libsamplerate project needs to be built before the pd plugin can be built

From the rtneural_pd directory.

```
cmake . -B build -DPD_PATH=<PATH TO PD SOURCE>
cmake --build build --target install
```

if you remove the -DPD_PATH variable , CMAKE will search for the pd source in the standard locations

a folder will be placed in the build directory with the rtneural and rtneural~ objects plus the help and examples. move this folder to a place in your pd path.

## Building the Max Plugin On Mac (or Linux)

Just like above, the libsamplerate project needs to be built before the Max plugin can be built

From the rtneural_max directory.

```
mkdir build
cd build
cmake ..
cmake --build .
```

After the mxo files are built, you will need to sign them locally:

```
codesign --force --deep -s - rtneural_max/externals/rtneural.mxo
codesign --force --deep -s - rtneural_max/externals/rtneural~.mxo
```

## Building the Max Plugin On PC

I have only been able to build this with GitHub actions using MinGW. See the build-max.yml file in the .github/workflows directory for a hope and a prayer to get this built

## Building the pd Plugin On PC

This should build just like the Mac plugin. It requires the bin directory from a built version of pd bin (bin55-2 is included) to exist inside the the rtneural_pd directory