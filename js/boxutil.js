BOX_UTIL = {
    disposeBox : (box) => {
        if (box != null && box.startPoint != null) {
            box.startEndTensor.dispose();
            box.startPoint.dispose();
            box.endPoint.dispose();
        }
    },
    createBox : (startEndTensor, startPoint, endPoint) => ({
        startEndTensor,
        startPoint: startPoint != null ? startPoint : tf.slice(startEndTensor, [0, 0], [-1, 2]),
        endPoint: endPoint != null ? endPoint : tf.slice(startEndTensor, [0, 2], [-1, 2])
    }),
    scaleBox : (box, factors) => {
        const starts = tf.mul(box.startPoint, factors);
        const ends = tf.mul(box.endPoint, factors);
        const newCoordinates = tf.concat2d([starts, ends], 1);
        return BOX_UTIL.createBox(newCoordinates);
    },
    getBoxSize : (box) => {
        return tf.tidy(() => {
            const diff = tf.sub(box.endPoint, box.startPoint);
            return tf.abs(diff);
        });
    },
    getBoxCenter : (box) => {
        return tf.tidy(() => {
            const halfSize = tf.div(tf.sub(box.endPoint, box.startPoint), 2);
            return tf.add(box.startPoint, halfSize);
        });
    },
    cutBoxFromImageAndResize: (box, image, cropSize) =>{
        const height = image.shape[1];
        const width = image.shape[2];
        const xyxy = box.startEndTensor;
        return tf.tidy(() => {
            const yxyx = tf.concat2d([
                xyxy.slice([0, 1], [-1, 1]), xyxy.slice([0, 0], [-1, 1]),
                xyxy.slice([0, 3], [-1, 1]), xyxy.slice([0, 2], [-1, 1])
            ], 0);
            const roundedCoords = tf.div(yxyx.transpose(), [height, width, height, width]);
            return tf.image.cropAndResize(image, roundedCoords, [0], cropSize, 'nearest');
        });
    },
    enlargeBox: (box, factor = 1.5) => {
        return tf.tidy(() => {
            const center = BOX_UTIL.getBoxCenter(box);
            const size = BOX_UTIL.getBoxSize(box);
            const newSize = tf.mul(tf.div(size, 2), factor);
            const newStart = tf.sub(center, newSize);
            const newEnd = tf.add(center, newSize);
            return BOX_UTIL.createBox(tf.concat2d([newStart, newEnd], 1), newStart, newEnd);
        });
    },

    cubeBox: (box, factor = 1.5 ) => {
        return tf.tidy(() => {
            const size = BOX_UTIL.getBoxSize(box);
            const center = BOX_UTIL.getBoxCenter(box);
            const halfSize = tf.div(size, 2);
            const newSize = tf.mul(halfSize.mean(), factor);
            const newStart = tf.sub(center, newSize);
            const newEnd = tf.add(center, newSize);
            return BOX_UTIL.createBox(tf.concat2d([newStart, newEnd], 1));
        });
    }

};