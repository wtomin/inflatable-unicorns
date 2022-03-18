const tf = require('@tensorflow/tfjs');

function save(epoch) {
  onEpochDone(
    epoch,
    new Float32Array(
      (
        window.net.faceFeatureExtractor
          ? Array.from(window.net.faceFeatureExtractor.serializeParams())
          : []
      )
        .concat(Array.from(window.net.serializeParams()))
    )
  )
}

async function train() {
  await load()

  for (let epoch = startEpoch; epoch < endEpoch; epoch++) {

    if (epoch !== startEpoch) {
      // ugly hack to wait for loss datas for that epoch to be resolved
      setTimeout(
        () => save(epoch - 1),
        10000
      )
    }
    window.lossValues[epoch] = 0

    const createBatches = window.createBatches || function(data, batchSize) {
      const shuffledInputs = prepareDataForEpoch(data)
      const batches = []
      for (let dataIdx = 0; dataIdx < shuffledInputs.length; dataIdx += batchSize) {
        batches.push(shuffledInputs.slice(dataIdx, dataIdx + batchSize))
      }
      return batches
    }

    const batches = createBatches(window.trainData, window.batchSize)
    console.log(batches)

    for (const [batchIdx, batchData] of batches.entries()) {
      const tsIter = Date.now()

      let bImages = await Promise.all(
        batchData
          .map(data => faceapi.fetchImage(getImageUri(data)))
      )

      let landmarks

      if (window.withRandomCrop || window.withFaceAlignment) {
        landmarks = await Promise.all(
          batchData
            .map(data => getLandmarksUri(data))
            .map(url => url ? faceapi.fetchJson(url) : null)
        )
      }

      if (window.withRandomCrop) {
        bImages = bImages.map((img, i) => {
          if (!landmarks[i]) {
            return img
          }

          const absoluteLandmarks = landmarks[i].map(pos => ({ x: pos.x * img.width, y: pos.y * img.height }))
          const { croppedImage } = randomCrop(img, absoluteLandmarks)

          return croppedImage
        })
      }

      if (window.withFaceAlignment) {
        bImages = await Promise.all(bImages.map(async (img, i) => {
          if (!landmarks[i]) {
            return img
          }

          const alignedRect = new faceapi.FaceLandmarks68(landmarks[i].map(({ x, y }) => new faceapi.Point(x, y)), img).align()
          const [alignedFace] = await faceapi.extractFaces(img, [alignedRect])
          return alignedFace
        }))
      }

      let tsBackward = Date.now()

      const netInput = await faceapi.toNetInput(bImages)
      const loss = computeLoss(netInput, batchData)

      tsBackward = Date.now() - tsBackward

      // start next iteration without waiting for loss data

      loss.data().then(data => {
        const lossValue = data[0]
        window.lossValues[epoch] += lossValue
        window.withLogging && log(`epoch ${epoch}, batchIdx ${batchIdx} - loss: ${lossValue}, ( ${window.lossValues[epoch]})`)
        loss.dispose()
        window.displayProgress && window.displayProgress(epoch, batchIdx, batches.length, window.lossValues[epoch])
      })

      window.withLogging && log(`epoch ${epoch}, batchIdx ${batchIdx} - backprop: ${tsBackward} ms, iter: ${Date.now() - tsIter} ms`)

      if (window.iterDelay) {
        await delay(window.iterDelay)
      } else {
        await tf.nextFrame()
      }
    }
  }
}

function CustomCCCMetric(x, y){
  /*def CCC_score(x, y):
vx = x - np.mean(x)
vy = y - np.mean(y)
rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
x_m = np.mean(x)
y_m = np.mean(y)
x_s = np.std(x)
y_s = np.std(y)
ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
return ccc*/
  return tf.tidy(() => {
  const xvalue = tf.moments(x,[0]);
  const yvalue = tf.moments(y, [0])
  const xm = xvalue.mean;
  const ym = yvalue.mean;
  const xs =xvalue.variance.sqrt();
  const ys = yvalue.variance.sqrt();
  const vx = x - xm;
  const vy = y - ym;
  const rho = tf.div(tf.mul(vx, vy).sum(), tf.mul(vx.square().sum().sqrt(), vy.square().sum().sqrt()).add(tf.backend().epsilon()))
  const ccc = tf.div(rho.mul(2).mul(xs).mul(ys), xvalue.variance.add(yvalue.variance).add(tf.sub(xm, ym).square()).add(tf.backend().epsilon()))
  return ccc 
  });
}

function CustomCCCLoss(x, y){
  return tf.tidy(() => {
    return tf.sub(1, CustomCCCMetric(x, y))
    });
}