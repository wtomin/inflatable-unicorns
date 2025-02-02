function getImageUri({ db, label, img }) {
  if (db === 'db') {
    const id = parseInt(img.replace('.jpg', ''))
    const dirNr = Math.floor(id / 5000)
    return `face-classification/cropped-faces/jpgs${dirNr + 1}/${img}`
  }
  if (db == 'arousal') {
    return `face-arousal-db/cropped_aligned/${img}`
  }
  if (db === 'kaggle') {
    return `kaggle-face-expressions-db/kaggle-face-expressions-db-cleaned/${label}/${img}`
  }

  if (db === 'db-augmented') {
    return `augmented/db/${label}/${img}`
  }

  if (db === 'kaggle-augmented') {
    return `augmented/kaggle/${label}/${img}`
  }

  if (db === 'jaffe') {
    return `jaffe-db/cropped-images/${img}`
  }

  if (db === 'kdef') {
    return `kdef-db/cropped-images/${img}`
  }

  if (db === 'ckplus') {
    return `ckplus-db/cropped-images/${img}`
  }

  if (db === 'imfdb') {
    return `imfdb/cropped-images/${img}`
  }

  if (db === 'actors-speech') {
    return `actors-db/speech/cropped-images/${img}`
  }

  if (db === 'actors-song') {
    return `actors-db/song/cropped-images/${img}`
  }

  throw new Error(`getImageUri - unknown db ${db}`)

}

function getLandmarksUri({ db, label, img }) {
  const nr = img.replace('.jpg', '')

  if (db === 'db' || db === 'db-augmented') {
    return `face-classification/face-landmarks/landmarks/${nr}.json`
  }

  if (db === 'jaffe') {
    return `jaffe-db/landmarks/${nr}.json`
  }

  if (db === 'kdef') {
    return `kdef-db/landmarks/${nr}.json`
  }

  if (db === 'ckplus') {
    return `ckplus-db/landmarks/${nr}.json`
  }

  if (db === 'actors-speech') {
    return `actors-db/speech/landmarks/${nr}.json`
  }

  if (db === 'actors-song') {
    return `actors-db/song/landmarks/${nr}.json`
  }

  return null

}

function prepareDataForEpoch(data) {
  return faceapi.shuffleArray(
    Object.keys(data).map(label => {
      const dataForLabel = data[label]//.map(data => ({ ...data, label }))
      let dataForLabelOut = dataForLabel

      if (label === 'vectorData') {
        return dataForLabelOut
      }

      for (let i = 0; i < 4; i++) {
        if (dataForLabelOut.length < 4000) {
          dataForLabelOut = dataForLabelOut.concat(faceapi.shuffleArray(dataForLabel))
        }
      }

      return dataForLabelOut//.slice(0, 4000)
    }).reduce((flat, arr) => arr.concat(flat))
  )
}


async function onEpochDone(epoch, params) {
  saveWeights(params || window.net, `face_expression_model_${epoch}.weights`)

  const numData = Object.values(window.trainData).reduce((sum, arr) => sum + arr.length, 0)
  const loss = window.lossValues[epoch]
  saveAs(new Blob([JSON.stringify({ loss, avgLoss: loss / numData })]), `face_expression_model_${epoch}.json`)

}

