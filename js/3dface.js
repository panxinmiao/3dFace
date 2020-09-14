const MESH_MODEL_INPUT_WIDTH = 120;
const MESH_MODEL_INPUT_HEIGHT = 120;
const UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD = 0.25;
//const UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD = 1;

const KEYPOINT_LANDMARKS_COUNT = 73;

class FaceMesh3D {
    constructor(blazeFace, faceChecker, meshModel, meshConfig, maxContinuousChecks = 5, detectionConfidence = 0.9, maxFaces = 10) {
        this.detectionConfidence = detectionConfidence;

        this.regionsOfInterest = [];
        this.runsWithoutFaceDetector = 0;
        this.boundingBoxDetector = blazeFace;
        this.faceChecker = faceChecker;

        this.meshDetector = meshModel;

        this.meshConfig = meshConfig;

        this.landmarksCount = meshConfig.w_shp.shape[0];

        this.coffCountShp = meshConfig.w_shp.shape[1];
        this.coffCountExp = meshConfig.w_exp.shape[1];

        this.meshWidth = MESH_MODEL_INPUT_WIDTH;
        this.meshHeight = MESH_MODEL_INPUT_HEIGHT;
        this.maxContinuousChecks = maxContinuousChecks;
        this.maxFaces = maxFaces;
    }

    async estimateFaces(input, returnTensors = false) {
        const image = tf.tidy(() => {
            if (!(input instanceof tf.Tensor)) {
                input = tf.browser.fromPixels(input);
            }
            return input.toFloat().expandDims(0);
        });
        const savedWebglPackDepthwiseConvFlag = tf.env().get('WEBGL_PACK_DEPTHWISECONV');
        tf.env().set('WEBGL_PACK_DEPTHWISECONV', true);
        const predictions = await this.predict(image);
        tf.env().set('WEBGL_PACK_DEPTHWISECONV', savedWebglPackDepthwiseConvFlag);
        image.dispose();

        if (predictions != null && predictions.length > 0) {
            return Promise.all(predictions.map(async (prediction, i) => {
                const { coords, scaledCoords, scaledCoordsFlatten, colorsFlatten, box, flag } = await prediction;
                let tensorsToRead = [flag];
                if (!returnTensors) {
                    tensorsToRead = tensorsToRead.concat([box.startPoint, box.endPoint, coords, scaledCoords, scaledCoordsFlatten, colorsFlatten]);
                }
                const tensorValues = await Promise.all(tensorsToRead.map(async (d) => d.array()));
                const flagValue = tensorValues[0];
                flag.dispose();
                if (flagValue < this.detectionConfidence) {
                    this.clearRegionOfInterest(i);
                }

                if(returnTensors){
                    return {
                        faceInViewConfidence: flagValue,
                        box: {
                            topLeft: box.startPoint.squeeze(),
                            bottomRight: box.endPoint.squeeze()
                        },
                        coords: coords,
                        scaledCoords: scaledCoords,
                        colorsFlatten: colorsFlatten,
                        scaledCoordsFlatten: scaledCoordsFlatten
                    }
                }else{
                    const [topLeft, bottomRight, coordsArr, coordsArrScaled, scaledCoordsArrFlatten, colorsArrFlatten] = tensorValues.slice(1);
                    coords.dispose();
                    scaledCoords.dispose();
                    scaledCoordsFlatten.dispose();
                    colorsFlatten.dispose();
                    return {
                        faceInViewConfidence: flagValue,
                        box: {
                            topLeft: topLeft[0],
                            bottomRight: bottomRight[0]
                        },
                        coords: coordsArr,
                        scaledCoords: coordsArrScaled,
                        colorsFlatten: colorsArrFlatten,
                        scaledCoordsFlatten: scaledCoordsArrFlatten
                    };
                }
                
            }));
        }
        return [];
    }

    async predict(input) {
        var shouldUpdateRegionsOfInterest = this.shouldUpdateRegionsOfInterest();
        if (shouldUpdateRegionsOfInterest) {
            const returnTensors = true;
            const annotateFace = false;
            const { boxes, scaleFactor } = await this.boundingBoxDetector.getBoundingBoxes(input, returnTensors, annotateFace);

            if (boxes.length === 0) {
                scaleFactor.dispose();
                this.clearAllRegionsOfInterest();
                return null;
            }
            const scaledBoxes = boxes.map((prediction) => BOX_UTIL.cubeBox(BOX_UTIL.scaleBox(prediction, scaleFactor)));
            boxes.forEach(BOX_UTIL.disposeBox);
            this.updateRegionsOfInterest(scaledBoxes);
            this.runsWithoutFaceDetector = 0;
        }
        else {
            this.runsWithoutFaceDetector++;
        }

        return tf.tidy(() => {
            return this.regionsOfInterest.map((box, i) => {

                const face = BOX_UTIL.cutBoxFromImageAndResize(box, input, [
                    this.meshHeight, this.meshWidth
                ]);

                const confidence = this.faceChecker.__predict(face);

                var coords = this.meshDetector.predict(face.reverse(-1));

                //coords.array().then( e => {console.log(e)});

                coords = coords.mul(this.meshConfig.std).add(this.meshConfig.mean);

                //p_ = tf.reshape(param[:, :12], (-1, 3, 4))
                //coords.shape [bs, 62]
                var p_ = coords.slice([0, 0], [-1, 12]);

                p_ = p_.reshape([-1, 3, 4]);

                /**
                 *      a, b, c, d          a,   b,   c,     d 
                 *      e, f, g, h   ===>  -e,  -f,  -g,   121-h
                 *      i, j, k, l          i,   j,   k,     l
                 */
  
                if(this.meshConfig.reverse){
                    p_ = p_.mul(tf.tensor2d([ [1, 1, 1, 1], [-1, -1, -1, -1], [1, 1, 1, 1] ])).add(tf.tensor2d([ [0, 0, 0, 0] , [0, 0, 0, 121], [0, 0, 0, 0] ]));
                }

                const p = p_.slice([0, 0, 0], [-1, -1, 3]);

                var offset = p_.slice([0, 0, 3], [-1, -1, 1]);

        
                var alpha_shp = coords.slice([0, 12], [-1, this.coffCountShp]).reshape([-1, this.coffCountShp, 1]);
                var alpha_exp = coords.slice([0, 12+this.coffCountShp], [-1, -1]).reshape([-1, this.coffCountExp, 1]);

                const bs = coords.shape[0];

                const w_shp = this.meshConfig.w_shp.broadcastTo([bs, this.landmarksCount, this.coffCountShp]);
                const w_exp = this.meshConfig.w_exp.broadcastTo([bs, this.landmarksCount, this.coffCountExp]);

                //vertex = u + tf.matmul(w_shp, alpha_shp) + tf.matmul(w_exp, alpha_exp)
                //vertex = tf.transpose(tf.reshape(vertex, (-1, self.resample_num+68, 3)), (0, 2, 1))
                //vertex = tf.matmul(p, vertex) + offset

                var vertex = this.meshConfig.u.add(w_shp.matMul(alpha_shp)).add(w_exp.matMul(alpha_exp));

                vertex = vertex.reshape([bs, -1, 3]).transpose([0, 2, 1]);

                vertex = p.matMul(vertex).add(offset);

                //var vertex = p.matMul( ( this.meshConfig.u.add(w_shp.matMul(alpha_shp)).add(w_exp.matMul(alpha_exp)) ).reshape([bs, -1, 3]).transpose([0, 2, 1]) ).add(offset);

                vertex = vertex.transpose([0, 2 ,1]);

                const coordsReshaped = tf.reshape(vertex, [-1, 3]);  //TODO batch size

                //BOX_UTIL.getBoxSize(box).div([this.meshWidth, this.meshHeight]);

                var normalizedBox = BOX_UTIL.getBoxSize(box).div([this.meshWidth, this.meshHeight]);
                // .concat(tf.tensor2d([1], [1, 1]), 1)
                //normalizedBox = normalizedBox.concat(normalizedBox.slice([0,1], [1,2]), 1);
                normalizedBox = normalizedBox.concat(normalizedBox.slice([0,1], [1,1]), 1);
                //const size = BOX_UTIL.getBoxSize(box)[0].div(this.meshWidth);

                //const scaleFactor = BOX_UTIL.getBoxSize(box).slice([0, 0], [1, 1]).reshape([-1]).div(this.meshWidth);

                const scaledCoords = coordsReshaped.mul(normalizedBox).add(box.startPoint.concat(tf.tensor2d([0], [1, 1]), 1));
                
                const scaledCoordsFlatten = scaledCoords.mul([-1, -1, 1]).reshape([-1]);

                const keypointIndices = this.meshConfig.keypoint;

                const keypoint = scaledCoords.gather(keypointIndices);

                
                const landmarksBox = this.calculateLandmarksBoundingBox(keypoint);
                
                const previousBox = this.regionsOfInterest[i];
                BOX_UTIL.disposeBox(previousBox);
                this.regionsOfInterest[i] = landmarksBox;

                let indices = scaledCoords.slice([0, 0], [-1, 2]);

                indices = indices.maximum([0,0]).minimum( [input.shape[1]-1,  input.shape[2]-1 ]);

                indices = tf.zeros([indices.shape[0], 1]).concat(indices, 1).toInt();

                const colorsFlatten = tf.gatherND(input.transpose([0, 2, 1, 3]), indices).reshape([-1]).div(255);

                const prediction = {
                    box: landmarksBox,
                    coords: vertex,
                    scaledCoords: scaledCoords,
                    scaledCoordsFlatten: scaledCoordsFlatten,
                    colorsFlatten: colorsFlatten,
                    flag: confidence.squeeze()
                };

                return prediction;

            });
        });
    }

    updateRegionsOfInterest(boxes) {
        for (let i = 0; i < boxes.length; i++) {
            const box = boxes[i];
            const previousBox = this.regionsOfInterest[i];
            let iou = 0;
            if (previousBox && previousBox.startPoint) {
                const [boxStartX, boxStartY, boxEndX, boxEndY] = box.startEndTensor.arraySync()[0];
                const [previousBoxStartX, previousBoxStartY, previousBoxEndX, previousBoxEndY] = previousBox.startEndTensor.arraySync()[0];
                const xStartMax = Math.max(boxStartX, previousBoxStartX);
                const yStartMax = Math.max(boxStartY, previousBoxStartY);
                const xEndMin = Math.min(boxEndX, previousBoxEndX);
                const yEndMin = Math.min(boxEndY, previousBoxEndY);
                const intersection = (xEndMin - xStartMax) * (yEndMin - yStartMax);
                const boxArea = (boxEndX - boxStartX) * (boxEndY - boxStartY);
                const previousBoxArea = (previousBoxEndX - previousBoxStartX) *
                    (previousBoxEndY - boxStartY);
                iou = intersection / (boxArea + previousBoxArea - intersection);
            }
            if (iou > UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD) {
                BOX_UTIL.disposeBox(box);
            }
            else {
                this.regionsOfInterest[i] = box;
                BOX_UTIL.disposeBox(previousBox);
            }
        }
        for (let i = boxes.length; i < this.regionsOfInterest.length; i++) {
            BOX_UTIL.disposeBox(this.regionsOfInterest[i]);
        }
        this.regionsOfInterest = this.regionsOfInterest.slice(0, boxes.length);
    }

    shouldUpdateRegionsOfInterest() {
        const roisCount = this.regionsOfInterest.length;
        const noROIs = roisCount === 0;
        if (this.maxFaces === 1 || noROIs) {
            return noROIs;
        }
        return roisCount !== this.maxFaces &&
            this.runsWithoutFaceDetector >= this.maxContinuousChecks;
    }

    clearRegionOfInterest(index) {
        if (this.regionsOfInterest[index] != null) {
            BOX_UTIL.disposeBox(this.regionsOfInterest[index]);
            this.regionsOfInterest = [
                ...this.regionsOfInterest.slice(0, index),
                ...this.regionsOfInterest.slice(index + 1)
            ];
        }
    }
    clearAllRegionsOfInterest() {
        for (let i = 0; i < this.regionsOfInterest.length; i++) {
            BOX_UTIL.disposeBox(this.regionsOfInterest[i]);
        }
        this.regionsOfInterest = [];
    }

    calculateLandmarksBoundingBox(landmarks) {
        const xs = landmarks.slice([0, 0], [KEYPOINT_LANDMARKS_COUNT, 1]);
        const ys = landmarks.slice([0, 1], [KEYPOINT_LANDMARKS_COUNT, 1]);
        const boxMinMax = tf.stack([xs.min(), ys.min(), xs.max(), ys.max()]);
        const box = BOX_UTIL.createBox(boxMinMax.expandDims(0));
        return BOX_UTIL.cubeBox(box);
    }

}