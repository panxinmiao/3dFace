class FaceExist {
    constructor(model, width, height) {
        this.graphModel = model;
        this.width = width;
        this.height = height;
        this.inputSizeData = [width, height];
        this.inputSize = tf.tensor1d([width, height]);
    }

    static async load(path, width, height){
        const checkFaceModel = await tf.loadGraphModel(path, {fromTFHub: true});
        return new FaceExist(checkFaceModel, width, height);
    }

    async checkFaces(inputImage, returnTensors) {
        const prediction = tf.tidy(() => {
            const resizedImage = inputImage.resizeNearestNeighbor([this.width, this.height]);
            const normalizedImage = resizedImage.sub(127.5).div(128).reverse(-1);
            const prediction = this.graphModel.predict(normalizedImage);
            return prediction;
        });

        if(!returnTensors){
            const vals = await prediction.array();
            prediction.dispose();
            return vals; 
        }else{
            return prediction;
        }

    }

    __predict(inputImage){
        const resizedImage = inputImage.resizeNearestNeighbor([this.width, this.height]);
        const normalizedImage = resizedImage.sub(127.5).div(128).reverse(-1);
        const prediction = this.graphModel.predict(normalizedImage);
        resizedImage.dispose();
        normalizedImage.dispose();
        return prediction;
    }

    async estimateFaces(input, returnTensors = false) {
        const [, width] = getInputTensorDimensions$1(input);
        const image = tf.tidy(() => {
            if (!(input instanceof tf.Tensor)) {
                input = tf.browser.fromPixels(input);
            }
            return input.toFloat().expandDims(0);
        });

        return checkFaces(image, returnTensors)
    }

}