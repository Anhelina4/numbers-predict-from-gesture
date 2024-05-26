let webcamElement = document.getElementById('camera')
let trainLabels = []
let trainXs
let trainYs
let mobilenet
let model
let array = Array.from(Array(10), () => 0)
let isPredicting = false

/**
 * If the video is wider than it is tall, adjust the width to match the height, and vice versa
 * @param width - The width of the video stream.
 * @param height - The height of the video stream.
 */
function adjustVideoSize(width, height) {
  const aspectRatio = width / height
  if (width >= height) {
    webcamElement.width = aspectRatio * webcamElement.height
  } else if (width < height) {
    webcamElement.height = webcamElement.width / aspectRatio
  }
}

/**
 * It takes a video stream from the webcam, and then it adds an event listener to the video element.
 *
 * The event listener waits for the video to load, and then it calls the adjustVideoSize function.
 *
 * The adjustVideoSize function is defined in the next code block.
 * @returns A promise that resolves to a stream object.
 */
async function setup() {
  return new Promise((resolve, reject) => {
    if (navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia({ video: { width: 224, height: 224 } })
        .then((stream) => {
          webcamElement.srcObject = stream
          webcamElement.addEventListener(
            'loadeddata',
            async () => {
              adjustVideoSize(
                webcamElement.videoWidth,
                webcamElement.videoHeight
              )
              resolve()
            },
            false
          )
        })
        .catch((error) => {
          reject(error)
        })
    } else {
      reject()
    }
  })
}

/**
 * It loads the MobileNet model from a URL, gets the output of the last convolutional layer, and
 * assigns it to the mobilenet variable
 */
async function loadMobilenet() {
  const mobileNetModel = await tf.loadLayersModel(
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json'
  )
  const layer = mobileNetModel.getLayer('conv_pw_13_relu')
  mobilenet = tf.model({ inputs: mobileNetModel.inputs, outputs: layer.output })
}

/**
 * It takes an image and returns a square image that is cropped from the center of the original image
 * @param img - The image to crop.
 * @returns A tensor with the same shape as the input tensor, but with the values from the height and
 * width dimensions cropped to be in the range [beginHeight, beginWidth, 0] to [size, size, 3].
 */
function cropImage(img) {
  const size = Math.min(img.shape[0], img.shape[1])
  const centerHeight = img.shape[0] / 2
  const centerWidth = img.shape[1] / 2
  const beginHeight = centerHeight - size / 2
  const beginWidth = centerWidth - size / 2
  return img.slice([beginHeight, beginWidth, 0], [size, size, 3])
}

/**
 * It takes a snapshot from the webcam, reverses the image, crops it, and normalizes it
 */
function capture() {
  return tf.tidy(() => {
    const webcamImage = tf.browser.fromPixels(webcamElement)
    const reversedImage = webcamImage.reverse(1)
    const croppedImage = cropImage(reversedImage)
    const batchedImage = croppedImage.expandDims(0)
    return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1))
  })
}

/**
 * It takes the trainLabels array and converts it into a one-hot encoded tensor.
 * @param numClasses - number of classes in the dataset
 */
function encodeLabels(numClasses) {
  for (let i = 0; i < trainLabels.length; i++) {
    const y = tf.tidy(() => {
      return tf.oneHot(tf.tensor1d([trainLabels[i]]).toInt(), numClasses)
    })
    if (trainYs == null) {
      trainYs = tf.keep(y)
    } else {
      const oldY = trainYs
      trainYs = tf.keep(oldY.concat(y, 0))
      oldY.dispose()
      y.dispose()
    }
  }
}

async function train() {
  try {
    trainYs = null
    /* Creating a model with 3 layers. The first layer is a flatten layer, which takes the output of the
        MobileNet model and flattens it into a vector. The second layer is a dense layer with 100 units,
        and the third layer is a dense layer with 10 units. The last layer is the output layer, and it
        uses a softmax activation function, which means it will return an array of 10 probability scores
        (summing to 1). Each score will be the probability that the current image belongs to one of our 10
        classes. */
    console.log('mobilenet-out', mobilenet.outputs[0])
    encodeLabels(10)
    model = tf.sequential({
      layers: [
        tf.layers.flatten({
          inputShape: mobilenet?.outputs[0]?.shape?.slice(1)
        }),
        tf.layers.dense({ units: 100, activation: 'relu' }),
        tf.layers.dense({ units: 10, activation: 'softmax' })
      ]
    })
    /* Training the model. */
    model.compile({
      optimizer: tf.train.adam(0.0001),
      loss: 'categoricalCrossentropy'
    })
    let loss = 0
    model.fit(trainXs, trainYs, {
      epochs: 10,
      callbacks: {
        onBatchEnd: async (batch, logs) => {
          loss = logs.loss.toFixed(5)
          // console.log('LOSS: ' + loss)
        }
      }
    })
  } catch (err) {
    console.log(err)
  }
}

function doTraining() {
  if (trainLabels.length !== 0) {
    train()
    $('#alert-success').addClass('show')
    $('#alert-error').removeClass('show')
    setTimeout(() => {
      $('#alert').removeClass('show')
    }, 3000)
  } else {
    console.log('click')
    $('#alert-error').addClass('show')
    $('#alert-success').removeClass('show')
    setTimeout(() => {
      $('#alert-error').removeClass('show')
    }, 3000)
  }
}

/**
 * It takes an example and a label, and adds them to the training data
 * @param example - a tensor of shape [1, 784]
 * @param label - The label for the example.
 */
function addExample(example, label) {
  if (trainXs == null) {
    trainXs = tf.keep(example)
  } else {
    const oldX = trainXs
    trainXs = tf.keep(oldX.concat(example, 0))
    oldX.dispose()
  }
  trainLabels.push(label)
}

/**
 * It takes a button element as an argument, and when the button is clicked, it will take 10 samples of
 * the webcam image, and add them to the training data
 * @param elem - the button that was clicked
 */
function handleButton(elem) {
  let count = 0
  console.log("elem", elem)
  const handle = () => {
    let label = parseInt(elem.id.split("_")[1])
    console.log("label", label)
    array[label]++
    document.getElementById('samples_' + elem.id).innerText = '' + array[label]
    const img = capture()
    addExample(mobilenet.predict(img), label)
  }
  const interval = setInterval(() => {
    handle()
    count++
    if (count >= 10) {
      clearInterval(interval)
    }
  }, 10)
}

/**
 * It captures an image from the webcam, runs it through the MobileNet model, and then runs the result
 * through the custom model
 */
async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = capture()
      const activation = mobilenet.predict(img)
      const predictions = model.predict(activation)
      return predictions.as1D().argMax()
    })
    document.getElementById('prediction').innerText = (
      await predictedClass.data()
    )[0]
    predictedClass.dispose()
    await tf.nextFrame()
  }
}

function setPredicting(predicting) {
  isPredicting = predicting
  predict()
}

async function init() {
  await setup()
  await loadMobilenet()
}

init()
