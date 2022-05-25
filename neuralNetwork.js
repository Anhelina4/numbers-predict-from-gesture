let webcamElement = document.getElementById('camera');
let labels = [];
let xs;
let ys;
let mobilenet;
let model;
let array = Array.from(Array(10), () => 0);
let isPredicting = false;

//================================= Web camera and MobileNet initialization =================================

// Adjust the size of the webcam viewport
// while maintaining the original aspect ratio
function adjustVideoSize(width, height) {
    const aspectRatio = width / height;
    if (width >= height) {
        webcamElement.width = aspectRatio * webcamElement.height;
    } else if (width < height) {
        webcamElement.height = webcamElement.width / aspectRatio;
    }
}

async function setup() {
    return new Promise((resolve, reject) => {
        // Prompt the user for permission to use webcam and
        // get the stream of media content from it
        if (navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia(
                {video: {width: 360, height: 250}}).then(stream => {
                webcamElement.srcObject = stream;
                webcamElement.addEventListener('loadeddata', async () => {
                    adjustVideoSize(
                        // Take the intrinsic height
                        // and width of the video track
                        webcamElement.videoWidth,
                        webcamElement.videoHeight);
                    resolve();
                }, false);
            }).catch(error => {
                reject(error);
            });
        } else {
            reject();
        }
    });
}

// Download and prepare MobileNet model
// The base model used in this example is MobileNet
// with a width of .25 and input image size of 224 X 224
async function loadMobilenet() {
    const mobileNetModel = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
    // Pick an intermediate depth wise convolutional layer
    const layer = mobileNetModel.getLayer('conv_pw_13_relu');
    mobilenet = tf.model({inputs: mobileNetModel.inputs, outputs: layer.output});
}

//================================= Frame capturing =========================================================

function cropImage(img) {
    const size = Math.min(img.shape[0], img.shape[1]);
    // Find the center of an image
    const centerHeight = img.shape[0] / 2;
    const centerWidth = img.shape[1] / 2;
    // Find new starting points for the cropped image
    const beginHeight = centerHeight - (size / 2);
    const beginWidth = centerWidth - (size / 2);
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
}

function capture() {
    // tf.tidy() executes the provided function and after
    // it is executed, cleans up all intermediate tensors
    // allocated by that function (except the returned ones)
    return tf.tidy(() => {
        // Create a tf.Tensor from an image
        const webcamImage = tf.browser.fromPixels(webcamElement);
        // Reverse image horizontally
        const reversedImage = webcamImage.reverse(1);
        // Crop image to a square with 3 channels (RGB)
        const croppedImage = cropImage(reversedImage);
        const batchedImage = croppedImage.expandDims(0);
        // Normalize image from 0:255 to -1:1
        return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
    });
}

//================================= Model training ==========================================================

function encodeLabels(numClasses) {
    for (let i = 0; i < labels.length; i++) {
        const y = tf.tidy(
            () => {
                return tf.oneHot(tf.tensor1d([labels[i]]).toInt(), numClasses)
            });
        if (ys == null) {
            // tf.keep() keeps a tf.Tensor generated inside
            // a tf.tidy() from being disposed automatically
            ys = tf.keep(y);
        } else {
            const oldY = ys;
            ys = tf.keep(oldY.concat(y, 0));
            // tf.dispose() disposes any tf.Tensors found
            // within the provided object
            oldY.dispose();
            y.dispose();
        }
    }
}


async function train() {
    ys = null;
    // Encode labels as OHE vectors
    encodeLabels(10);
    model = tf.sequential({
        layers: [
            // Simply take the output of the last layer
            // of our truncated MobileNet model and flatten it
            tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
            // Then pass the result to the dense layer - the 'core'
            // of our second fine-tuning model
            tf.layers.dense({units: 100, activation: 'relu'}),
            // Output layer gives us probabilities for each
            // of the output classes
            tf.layers.dense({units: 10, activation: 'softmax'})
        ]
    });

    // Compile the fine-tuning model using Adam optimizer
    // and categorical crossentropy loss function
    model.compile({optimizer: tf.train.adam(0.0001), loss: 'categoricalCrossentropy'});
    let loss = 0;
    // Train the model for 10 epochs and report
    // loss value after each epoch
    model.fit(xs, ys, {
        epochs: 10,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                loss = logs.loss.toFixed(5);
                console.log('LOSS: ' + loss);
            }
        }
    });
}

// This function is called when user clicks the Train button
function doTraining() {
    train();
    alert("Training Done!")
}

//================================= Creating one training sample ============================================

function addExample(example, label) {
    if (xs == null) {
        xs = tf.keep(example);
    } else {
        const oldX = xs;
        xs = tf.keep(oldX.concat(example, 0));
        oldX.dispose();
    }
    labels.push(label);
}

// This function is called when user clicks one of the label buttons
function handleButton(elem) {
    // Get the ground-truth label by the id
    // of the button that the user clicked
     let count=0
     
     const handle=()=>{
        let label = parseInt(elem.id);
        array[label]++;
        // Update the according counter
        document.getElementById("samples_" + elem.id).innerText = "" + array[label];
        // Capture an image from the webcam feed
        const img = capture();
        // And pass it to the MobileNet model, then save its output
        addExample(mobilenet.predict(img), label);
     }

     let interval = setInterval(()=>{
        count++
        if(count>=10) clearInterval(interval)
        handle()
    }, 10)
    
}

//================================= Inference process =======================================================

// The process of recognizing digits is quite similar to the training process
// itself - take an image from a webcam stream, run it through the MobileNet model,
// take the output, pass it to the trained fine-tuning model
// and take the value with maximum probability
async function predict() {
    while (isPredicting) {
        const predictedClass = tf.tidy(() => {
            const img = capture();
            const activation = mobilenet.predict(img);
            const predictions = model.predict(activation);
            return predictions.as1D().argMax();
        });
        document.getElementById("prediction").innerText = (await predictedClass.data())[0];
        predictedClass.dispose();
        await tf.nextFrame();
    }
}

// This function is called when user clicks
// the Start/Stop Predicting buttons
function setPredicting(predicting) {
    isPredicting = predicting;
    predict();
}

//================================= Aux functions ===========================================================

function saveModel() {
    model.save('downloads://my_model');
}

async function init() {
    await setup();
    await loadMobilenet();
}

init();