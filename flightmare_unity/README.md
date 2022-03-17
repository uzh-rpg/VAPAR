# Flightmare Unity (customized version)

For general information on the Flightmare Unity application, see the parent repository of this one, as well as the documentation. This repository contains some changes used for my Master's thesis and a following project on optical flow. It accompanies a [customized version](https://github.com/swengeler/flightmare) of the Flightmare simulator.

The main difference of this customized version compared to the original one (as mentioned in the respective Flightmare repository), are the changes to the handling of optical flow data.

## Building the binary

While the latest version of the resulting Flightmare Unity binary is included under `flightmare_unity.tar.gz`, it would have to be recompiled if any additional changes are made. For instructions on how to do this see the original Flightmare documentation.

## CLI parameters

The following command line parameters can be specified beyond those that can be specified for all Unity applications:

- `-dc-timeout` the disconnect timeout in seconds for communication between Flightmare itself and this Flightmare Unity application (after timing out, Unity will reset the view and new connections can be accepted)
- `-input-port` corresponds to the Flightmare publication port (`pub_port`) and should match between the client and server; `10253` by default
- `-output-port` corresponds to the Flightmare subscription port (`sub_port`) and should match between the client and server; `10254` by default

In addition, one useful parameter for all Unity applications is `-batchmode`, which disables actually drawing anything visible on screen. Images are still rendered correctly.

## Running this application headless on a server (specifically the RPG snaga server)

These are just some short "instructions" for how I managed to run the Flightmare Unity application on a display-less server. Due to some changes in the environment this sometimes did not work (I suspect because no X-server was running), but for the most part this is probably the best way of going about it.

First of all, TurboVNC needs to be installed (might be in a different location) to be able to render images on a virtual display. It can then be started using the following command (with the installation on the snaga server). 

```shell
/opt/TurboVNC/bin/vncserver :0
```

According to my undestanding, here `:0` is the ID of the (virtual?) display that will be started up, not e.g. a device ID for which GPU to run everything on (see [here](https://www.commandlinux.com/man-page/man1/vncserver.1.html) for more information). I have not been able to run the command below (and thus the Flightmare Unity application) with anything but display `:0` for some reason, but I think that this should work in principle.

Running the next to command then needs to happen in the same terminal/session as the above (I would recommend using tmux for this). The `DISPLAY` should be the one specified in the above command, and the same should be the case for the first number in `:0.0`. The second number should in principle select the GPU to use (see [here](https://wiki.archlinux.org/title/VirtualGL#Running_applications) for more information), but I again could not get this to work for anything other than `0`. This means that GPU 0 has to be used (unless whoever uses this in the future does not encounter these problems or can fix them), which can be inconvenient, especially if other people are using the server. It is **essential** to use the `-batchmode` option, otherwise this will not work; apart from that any other CLI parameters can be specified:

```shell
DISPLAY=:0 vglrun -v -d :0.0 ./flightmare_unity.x86_64 -batchmode
```