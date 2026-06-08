// ─── Global Variables ───────────────────────────────────────────────────────

let webcamElement = document.getElementById('camera') // Reference to the <video> element in the DOM
let trainLabels = []   // Array storing integer labels for each training example
let trainXs         // Tensor holding all training feature vectors (MobileNet embeddings)
let trainYs         // Tensor holding one-hot encoded training labels
let mobilenet       // Truncated MobileNet model used as a feature extractor
let model           // Custom classification head trained on top of MobileNet features
let array = Array.from(Array(10), () => 0)      // Per-class training sample counters [0..9]
let arrayTest = Array.from(Array(10), () => 0)  // Per-class test sample counters [0..9]
let isPredicting = false  // Flag that controls the real-time prediction loop

// Separate storage for test (evaluation) data
let testLabels = []  // Integer labels for test examples
let testXs           // Tensor holding test feature vectors
let testYs           // Tensor holding one-hot encoded test labels

// ─── Webcam Setup ────────────────────────────────────────────────────────────

/**
 * Ensures the webcam element's displayed dimensions maintain the correct
 * aspect ratio regardless of whether the stream is landscape or portrait.
 *
 * @param {number} width  - Native width of the incoming video stream.
 * @param {number} height - Native height of the incoming video stream.
 */
function adjustVideoSize(width, height) {
  const aspectRatio = width / height
  if (width >= height) {
    // Landscape or square: scale width to match fixed height
    webcamElement.width = aspectRatio * webcamElement.height
  } else {
    // Portrait: scale height to match fixed width
    webcamElement.height = webcamElement.width / aspectRatio
  }
}

/**
 * Requests webcam access, attaches the stream to the video element,
 * and waits until the video metadata is ready before resolving.
 * The video is requested at 224×224 px — the input size expected by MobileNet.
 *
 * @returns {Promise<void>} Resolves when the webcam feed is ready to use.
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
              adjustVideoSize(webcamElement.videoWidth, webcamElement.videoHeight)
              resolve()
            },
            false
          )
        })
        .catch((error) => reject(error))
    } else {
      reject(new Error('getUserMedia is not supported in this browser.'))
    }
  })
}

// ─── MobileNet Feature Extractor ─────────────────────────────────────────────

/**
 * Downloads MobileNet V1 from Google's CDN and creates a truncated version
 * that outputs feature maps from the last convolutional layer ('conv_pw_13_relu')
 * instead of final class probabilities.
 *
 * These feature maps serve as rich visual embeddings that our lightweight
 * classifier is trained on top of (transfer learning).
 */
async function loadMobilenet() {
  const mobileNetModel = await tf.loadLayersModel(
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json'
  )
  // Cut the network at the last conv layer to get a 7×7×256 feature map
  const layer = mobileNetModel.getLayer('conv_pw_13_relu')
  mobilenet = tf.model({ inputs: mobileNetModel.inputs, outputs: layer.output })
}

// ─── Image Preprocessing ─────────────────────────────────────────────────────

/**
 * Crops a center square from the given image tensor.
 * This avoids distortion when the webcam stream is not perfectly square.
 *
 * @param {tf.Tensor3D} img - Raw image tensor of shape [H, W, 3].
 * @returns {tf.Tensor3D}   - Square tensor of shape [min(H,W), min(H,W), 3].
 */
function cropImage(img) {
  const size = Math.min(img.shape[0], img.shape[1])
  const centerHeight = img.shape[0] / 2
  const centerWidth  = img.shape[1] / 2
  const beginHeight  = centerHeight - size / 2
  const beginWidth   = centerWidth  - size / 2
  return img.slice([beginHeight, beginWidth, 0], [size, size, 3])
}

/**
 * Grabs a single frame from the webcam, mirrors it horizontally (so the image
 * acts like a mirror for the user), center-crops it to a square, and normalises
 * pixel values from [0, 255] to [-1, 1] as required by MobileNet.
 *
 * @returns {tf.Tensor4D} Preprocessed image tensor of shape [1, 224, 224, 3].
 */
function capture() {
  return tf.tidy(() => {
    const webcamImage   = tf.browser.fromPixels(webcamElement) // [H, W, 3], uint8
    const reversedImage = webcamImage.reverse(1)               // Mirror along width axis
    const croppedImage  = cropImage(reversedImage)             // Square crop
    const batchedImage  = croppedImage.expandDims(0)           // Add batch dim → [1, H, W, 3]
    // Normalise to [-1, 1]: pixel / 127 - 1
    return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1))
  })
}

// ─── Label Encoding ──────────────────────────────────────────────────────────

/**
 * Converts the integer labels in `trainLabels` into a single stacked one-hot
 * tensor stored in `trainYs`.
 * Example: label 3 with numClasses=10 → [0,0,0,1,0,0,0,0,0,0]
 *
 * @param {number} numClasses - Total number of gesture classes (10 for digits 0–9).
 */
function encodeLabels(numClasses) {
  for (let i = 0; i < trainLabels.length; i++) {
    const y = tf.tidy(() =>
      tf.oneHot(tf.tensor1d([trainLabels[i]]).toInt(), numClasses)
    )
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

/**
 * Same as `encodeLabels` but operates on `testLabels` / `testYs`.
 * Kept separate so training and evaluation data remain independent.
 *
 * @param {number} numClasses - Total number of gesture classes.
 */
function encodeTestLabels(numClasses) {
  for (let i = 0; i < testLabels.length; i++) {
    const y = tf.tidy(() =>
      tf.oneHot(tf.tensor1d([testLabels[i]]).toInt(), numClasses)
    )
    if (testYs == null) {
      testYs = tf.keep(y)
    } else {
      const oldY = testYs
      testYs = tf.keep(oldY.concat(y, 0))
      oldY.dispose()
      y.dispose()
    }
  }
}

// ─── Model Definition & Training ─────────────────────────────────────────────

/**
 * Builds and trains a small fully-connected classifier on top of the MobileNet
 * feature embeddings:
 *
 *   Flatten → Dense(100, ReLU) → Dense(10, Softmax)
 *
 * Architecture reasoning:
 *   - Flatten: converts the 3-D feature map from MobileNet into a 1-D vector.
 *   - Dense(100, ReLU): learns non-linear combinations of the features.
 *   - Dense(10, Softmax): produces a probability distribution over the 10 digit classes.
 *
 * After training completes the model is evaluated automatically via `evaluateModel()`.
 */
async function train() {
  try {
    trainYs = null // Reset label tensor before re-encoding

    // Encode integer labels → one-hot tensors for both splits
    encodeLabels(10)
    encodeTestLabels(10)

    // Build the sequential classifier head
    model = tf.sequential({
      layers: [
        tf.layers.flatten({ inputShape: mobilenet?.outputs[0]?.shape?.slice(1) }),
        tf.layers.dense({ units: 100, activation: 'relu' }),
        tf.layers.dense({ units: 10,  activation: 'softmax' })
      ]
    })

    // Adam with a small learning rate works well for fine-tuning on limited data
    model.compile({
      optimizer: tf.train.adam(0.0001),
      loss: 'categoricalCrossentropy'
    })

    model.fit(trainXs, trainYs, {
      epochs: 10,
      callbacks: {
        onBatchEnd: async (batch, logs) => {
          // Hook available for real-time loss logging if needed
        },
        onTrainEnd: async () => {
          // Automatically evaluate the model on the held-out test set after training
          await evaluateModel()
        }
      }
    })
  } catch (err) {
    console.log(err)
  }
}

/**
 * Called when the user clicks the "Train" button.
 * Guards against training with no data and shows brief UI feedback alerts.
 */
function doTraining() {
  if (trainLabels.length !== 0) {
    train()
    $('#alert-success').addClass('show')
    $('#alert-error').removeClass('show')
    setTimeout(() => $('#alert').removeClass('show'), 3000)
  } else {
    $('#alert-error').addClass('show')
    $('#alert-success').removeClass('show')
    setTimeout(() => $('#alert-error').removeClass('show'), 3000)
  }
}

// ─── Data Collection ─────────────────────────────────────────────────────────

/**
 * Appends a single MobileNet embedding to the training feature tensor
 * and records its corresponding integer label.
 *
 * @param {tf.Tensor} example - MobileNet output tensor for one frame.
 * @param {number}    label   - Integer class label (0–9).
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
 * Appends a single MobileNet embedding to the test feature tensor
 * and records its corresponding integer label.
 * Test data is never used during training — only for evaluation.
 *
 * @param {tf.Tensor} example - MobileNet output tensor for one frame.
 * @param {number}    label   - Integer class label (0–9).
 */
function addTestExample(example, label) {
  if (testXs == null) {
    testXs = tf.keep(example)
  } else {
    const oldX = testXs
    testXs = tf.keep(oldX.concat(example, 0))
    oldX.dispose()
  }
  testLabels.push(label)
}

// ─── Evaluation Metrics ──────────────────────────────────────────────────────

/**
 * Computes overall classification accuracy.
 *
 * Accuracy = (number of correct predictions) / (total predictions)
 *
 * @param {tf.Tensor1D} trueClasses      - Ground-truth class indices.
 * @param {tf.Tensor1D} predictedClasses - Predicted class indices.
 * @returns {Promise<number>} Accuracy in the range [0, 1].
 */
async function calculateAccuracy(trueClasses, predictedClasses) {
  const accuracyTensor = tf.equal(trueClasses, predictedClasses)
    .sum()
    .div(trueClasses.shape[0])
  const accuracy = await accuracyTensor.data()
  accuracyTensor.dispose()
  return accuracy[0]
}

/**
 * Computes binary precision and recall for class 1.
 * Note: for a 10-class problem you would normally compute these per-class
 * and macro-average them; this implementation treats it as a binary problem.
 *
 * Precision = TP / (TP + FP)  — "of all predicted positives, how many were right?"
 * Recall    = TP / (TP + FN)  — "of all actual positives, how many did we catch?"
 *
 * @param {tf.Tensor1D} trueClasses      - Ground-truth class indices.
 * @param {tf.Tensor1D} predictedClasses - Predicted class indices.
 * @returns {{ precision: number, recall: number }}
 */
function calculatePrecisionRecall(trueClasses, predictedClasses) {
  // True Positives: both ground truth and prediction equal 1
  const truePositive    = tf.logicalAnd(
    tf.equal(trueClasses, 1),
    tf.equal(predictedClasses, 1)
  ).sum().dataSync()[0]

  const predictedPositive = tf.equal(predictedClasses, 1).sum().dataSync()[0] // TP + FP
  const actualPositive    = tf.equal(trueClasses, 1).sum().dataSync()[0]      // TP + FN

  const precision = truePositive / predictedPositive
  const recall    = truePositive / actualPositive

  return { precision, recall }
}

/**
 * Builds and logs a 10×10 confusion matrix to the browser console.
 *
 * Rows  = actual (true) class
 * Cols  = predicted class
 * Diagonal cells = correct predictions; off-diagonal = classification errors.
 *
 * This helps identify which digit pairs the model confuses most often.
 */
async function buildConfusionMatrix() {
  const predictions      = model.predict(testXs)
  const predictedClasses = predictions.argMax(-1).dataSync() // Predicted class per sample
  const trueClasses      = testYs.argMax(-1).dataSync()      // True class per sample

  const numClasses = 10
  // Initialise a 10×10 matrix of zeros
  const confusionMatrix = Array.from(Array(numClasses), () => Array(numClasses).fill(0))

  for (let i = 0; i < trueClasses.length; i++) {
    confusionMatrix[trueClasses[i]][predictedClasses[i]]++
  }

  console.table(confusionMatrix)
}

/**
 * Runs a full evaluation pass on the held-out test set after training:
 *   1. Computes overall accuracy.
 *   2. Computes binary precision and recall.
 *   3. Prints the confusion matrix.
 * All results are logged to the browser console.
 */
async function evaluateModel() {
  try {
    const predictions      = model.predict(testXs)
    const predictedClasses = predictions.argMax(-1)
    const trueClasses      = testYs.argMax(-1)

    // 1. Accuracy
    const accuracy = await calculateAccuracy(trueClasses, predictedClasses)
    console.log(`Accuracy: ${accuracy}`)

    // 2. Precision & Recall
    const { precision, recall } = calculatePrecisionRecall(trueClasses, predictedClasses)
    console.log(`Precision: ${precision}`)
    console.log(`Recall:    ${recall}`)

    // 3. Confusion Matrix
    buildConfusionMatrix()
    console.log('---------------------------------------')

    // Clean up intermediate tensors to free GPU/CPU memory
    predictions.dispose()
    predictedClasses.dispose()
    trueClasses.dispose()
  } catch (err) {
    console.log(err)
  }
}

// ─── Button Handlers ─────────────────────────────────────────────────────────

/**
 * Collects 20 test samples for the digit class encoded in the button's id
 * (e.g. id="test_3" → label 3). Frames are captured every 10 ms.
 * The sample count displayed next to the button is updated after each capture.
 *
 * @param {HTMLElement} elem - The "Test" button element that was pressed.
 */
function handleTestButton(elem) {
  let count = 0
  const handle = () => {
    const label = parseInt(elem.id.split('_')[1])
    arrayTest[label]++
    document.getElementById('samples_' + elem.id).innerText = '' + arrayTest[label]
    const img = capture()
    addTestExample(mobilenet.predict(img), label)
  }
  const interval = setInterval(() => {
    handle()
    if (++count >= 20) clearInterval(interval)
  }, 10)
}

/**
 * Collects 20 training samples for the digit class encoded in the button's id
 * (e.g. id="train_5" → label 5). Frames are captured every 10 ms.
 * The sample count displayed next to the button is updated after each capture.
 *
 * @param {HTMLElement} elem - The "Train" button element that was pressed.
 */
function handleTrainButton(elem) {
  let count = 0
  const handle = () => {
    const label = parseInt(elem.id.split('_')[1])
    array[label]++
    document.getElementById('samples_' + elem.id).innerText = '' + array[label]
    const img = capture()
    addExample(mobilenet.predict(img), label)
  }
  const interval = setInterval(() => {
    handle()
    if (++count >= 20) clearInterval(interval)
  }, 10)
}

// ─── Real-time Prediction Loop ────────────────────────────────────────────────

/**
 * Continuously captures webcam frames and displays the predicted digit in the
 * element with id="prediction". Runs as long as `isPredicting` is true.
 *
 * Uses `tf.nextFrame()` between iterations so the browser UI thread stays
 * responsive and does not freeze.
 */
async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img        = capture()
      const activation = mobilenet.predict(img)   // Extract MobileNet features
      const predictions = model.predict(activation) // Run classifier head
      return predictions.as1D().argMax()           // Pick the highest-probability class
    })
    document.getElementById('prediction').innerText = (await predictedClass.data())[0]
    predictedClass.dispose()
    await tf.nextFrame() // Yield to the browser between frames
  }
}

/**
 * Toggles real-time prediction on or off and starts the prediction loop.
 *
 * @param {boolean} predicting - Pass `true` to start predicting, `false` to stop.
 */
function setPredicting(predicting) {
  isPredicting = predicting
  predict()
}

// ─── Initialisation ───────────────────────────────────────────────────────────

/**
 * Entry point: sets up the webcam stream and loads the MobileNet feature extractor.
 * Called automatically when the page loads.
 */
async function init() {
  await setup()        // Start webcam
  await loadMobilenet() // Load pretrained MobileNet
}

init()