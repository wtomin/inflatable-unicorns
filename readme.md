# The Light-Weight Arousal Model

This is a repository forked from https://github.com/justadudewhohacks/inflatable-unicorns, which was created by the author of the [face-api.js](https://justadudewhohacks.github.io/face-api.js/docs/index.html).

We modified the training script and validation script so that we can use it to train a light-weight CNN for arousal prediction.

Similar to the `faceExpression` CNN used in the [face-api.js](https://justadudewhohacks.github.io/face-api.js/docs/index.html), this light-weight arousal model uses a shallow mobilenet as the backbone. The output dimension is one (arousal score).

# Dependencies

## Nodejs

Firstly, install `nvm` which allows you to quickly install and use different versions of node via the command line. `nvm` can be installed with:

```
wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
```

To check that nvm is installed, type `nvm --version`.

Next, let's install `Nodejs` version 14.4.

Simply run `nvm install 14.4.0`. Check its installation with `node --version`. If successful, you will see `v14.4.0`.

Run the command below to tell Ubuntu that we want to install the Nodejs package from nodesource.
```
curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -
```

Once we're done setting up Nodesource, we can now install Nodejs.
Run `sudo apt-get install -y nodejs`.

Simply type `nodejs -v` into your terminal and it should return v14.x version.

You should have npm automatically installed at this point. To check what npm version you have, run `npm version`.

More details about installation can be found [here](https://www.freecodecamp.org/news/how-to-install-node-js-on-ubuntu-and-update-npm-to-the-latest-version/).


## opencv4nodejs

`opencv4nodejs` is the opencv nodejs binding used by the author Vincent Mühler. I recommend you to use the docker image provided by the author. [How to use it](https://github.com/justadudewhohacks/opencv4nodejs#usage-with-docker). If you have not installed docker, please follow the [official installation instruction](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) on Ubuntu. Besides, install `NVIDIA-DOCKER` from this [official installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

Pull the image with
```
sudo docker pull justadudewhohacks/opencv-nodejs
```
## face-api.js

The `face-api.js` is not the same as the original implementation, but my forked repository. Run 

```
git clone https://github.com/wtomin/face-api.js
```

And store the face-api.js in the same directory as the `inflatable-unicorns`. Then in the `face-api.js`, switch to another branch with `git checkout arousalNet`. 

# Data Preparation

Download the `cropped_aligned.zip` with
```
# cropped_aligned.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1mJdgQviC5i2IFlmMKFkZwG8_RwHFSRrX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1mJdgQviC5i2IFlmMKFkZwG8_RwHFSRrX" -O cropped_aligned.zip && rm -rf /tmp/cookies.txt
```

Move the `cropped_aligned.zip` into `train/faceArousal/public/face-arousal-db` and extract it. Afterwards, you can delete the zip file.

# Run a Container Interactively

Start a container with 
```
docker run -it -v the-parent-directory-of-WORKDIR-on-host-machine:/tmp -w /tmp --gpus all -p 8000:8000 -p 9229:9229 justadudewhohacks/opencv-nodejs bash
```
For example, my `the-parent-directory-of-WORKDIR-on-host-machine` is `/home/ddeng`.

After entering the container, the current working directory is `tmp`, which is the same as your WORKDIR in your host machine.

```
$tmp/inflatable-unicorns/: ls
common           node_modules       package.json  readme.md  tsconfig.json
cropped_aligned  package-lock.json  quantize      train
```

In the current directory, run `npm install` to install the packages. In addition, install the typescript globally by `npm install typescript -g`.

Next, run `export NODE_PATH=/usr/lib/node_modules` to help node find the installed opencv4nodejs module.

# Start the server

In `train/faceArousal/public`, there is a typescript file `server.ts`. Firstly, we need to compile it to javascript, using:

```
tsc server.ts --target es5
```

This generates a file named `server.js` in the current directory. 

Then, we can start the server by running `node server.js` in the container. If you see the following text on the terminal
```
Listening on port 8000!
```
, you should go to the `localhost:8000` with your browser (I recommend Chrome). It automatically jumps to `localhost:8000/train`, where the training function is called.

Press `ctrl + shift + j` in Chrome. It opens the console where the batch index, the epoch index are printed continuously. 

## Training Configurations

In the `face/Arousal/train.html`, we define the training configurations, such as:
```
    const startEpoch = 0 
    const endEpoch = 1 
    const learningRate = 0.001
    window.optimizer = tf.train.adam(learningRate, 0.9, 0.999, 1e-8)
    window.batchSize = 32

    loss = tf.losses.meanSquaredError // or loss = CustomCCCLoss

```








