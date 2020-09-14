const MESH_MODEL_INPUT_WIDTH = 112;
const MESH_MODEL_INPUT_HEIGHT = 112;
const UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD = 0.25;
//const UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD = 1;

const LANDMARKS_COUNT = 106;

class FaceMesh2D {
    constructor(blazeFace, faceChecker, meshModel,  maxFaces = 10, maxContinuousChecks = 5, detectionConfidence = 0.9) {
        //this.pipeline = new Pipeline(blazeFace, blazeMeshModel, MESH_MODEL_INPUT_WIDTH, MESH_MODEL_INPUT_HEIGHT, maxContinuousChecks, maxFaces);
        this.detectionConfidence = detectionConfidence;

        this.regionsOfInterest = [];
        this.runsWithoutFaceDetector = 0;
        this.boundingBoxDetector = blazeFace;
        this.faceChecker = faceChecker;

        this.meshDetector = meshModel;
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
                const { coords, scaledCoords, box, flag } = await prediction;

                let tensorsToRead = [flag];
                if (!returnTensors) {
                    tensorsToRead = tensorsToRead.concat([box.startPoint, box.endPoint, coords, scaledCoords]);
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
                    }
                }else{
                    const [topLeft, bottomRight, coordsArr, coordsArrScaled] = tensorValues.slice(1);;
                    coords.dispose();
                    scaledCoords.dispose();
                    return {
                        faceInViewConfidence: flagValue,
                        box: {
                            topLeft: topLeft[0],
                            bottomRight: bottomRight[0]
                        },
                        coords: coordsArr,
                        scaledCoords: coordsArrScaled
                        
                    };
                }
                
            }));
        }
        return [];
    }


    async predict(input) {
        if (this.shouldUpdateRegionsOfInterest()) {
            const returnTensors = true;
            const annotateFace = false;
            const { boxes, scaleFactor } = await this.boundingBoxDetector.getBoundingBoxes(input, returnTensors, annotateFace);

            if (boxes.length === 0) {
                scaleFactor.dispose();
                this.clearAllRegionsOfInterest();
                return null;
            }
            const scaledBoxes = boxes.map((prediction) => BOX_UTIL.cubeBox(BOX_UTIL.scaleBox(prediction, scaleFactor)), 1.4);

            boxes.forEach(BOX_UTIL.disposeBox);

            this.updateRegionsOfInterest(scaledBoxes);
            this.runsWithoutFaceDetector = 0;
        }
        else {
            this.runsWithoutFaceDetector++;
        }
        return tf.tidy( () => {
            return this.regionsOfInterest.map((box, i) => {

                const face = BOX_UTIL.cutBoxFromImageAndResize(box, input, [
                    this.meshHeight, this.meshWidth
                ]);

                const confidence = this.faceChecker.__predict(face);

                const coords = this.meshDetector.predict(face.div(255));
                const coordsReshaped = tf.reshape(coords, [-1, 2]);
                const scaledCoords = tf.mul(coordsReshaped, BOX_UTIL.getBoxSize(box)).add(box.startPoint);
                const landmarksBox = this.calculateLandmarksBoundingBox(scaledCoords);
    
                const previousBox = this.regionsOfInterest[i];
                BOX_UTIL.disposeBox(previousBox);
                this.regionsOfInterest[i] = landmarksBox;

                const prediction = {
                    box: landmarksBox,
                    coords: coords,
                    scaledCoords: scaledCoords,
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
        const xs = landmarks.slice([0, 0], [LANDMARKS_COUNT, 1]);
        const ys = landmarks.slice([0, 1], [LANDMARKS_COUNT, 1]);
        const boxMinMax = tf.stack([xs.min(), ys.min(), xs.max(), ys.max()]);
        const box = BOX_UTIL.createBox(boxMinMax.expandDims(0));
        return BOX_UTIL.cubeBox(box);
    }

}