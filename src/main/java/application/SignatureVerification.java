package global.skymind.GroupProject;

import ch.qos.logback.classic.BasicConfigurator;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.VGG19;
import org.nd4j.common.primitives.Pair;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.PathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.bytedeco.opencv.global.opencv_imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


public class SignatureVerification {

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(SignatureVerification.class);

    static Random rand = new Random();
    static String[] allowedFormats = BaseImageLoader.ALLOWED_FORMATS;
    static PathLabelGenerator labelMaker = new ParentPathLabelGenerator();

    static int width = 300;
    static int height = 300;
    static int seed = 123;
    static int batchSize = 5;
    static double learningRate = 0.0001;
    static int epoch = 4;

    public static void main(String[] args) throws IOException {

        gui();

    }

    public static void gui() {

        JFileChooser fileChooser = new JFileChooser();

        JFrame window = new JFrame("Signature Verifier");
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        window.setSize(700, 500);

        JMenuBar menu = new JMenuBar();
        JMenu file = new JMenu("File");
        menu.add(file);
        JMenuItem browse = new JMenuItem("Browse");

        ImagePanel imagePanel = new ImagePanel();
        JButton verify = new JButton("Verify");
        JLabel result = new JLabel("-");

        JPanel bottomPanel = new JPanel();

        bottomPanel.add(verify);
        bottomPanel.add(result);
        browse.addActionListener(e -> {

            int val = fileChooser.showOpenDialog(window);

            if (val == JFileChooser.APPROVE_OPTION) {
                File readFile = fileChooser.getSelectedFile();

                try {
                    imagePanel.repaint();
                    imagePanel.readImage(readFile);

                } catch (IOException ioException) {
                    ioException.printStackTrace();
                }
            }

        });

        file.add(browse);

        verify.addActionListener(e -> {

            try {

                boolean prediction;
                prediction = predict(imagePanel.getImage());

                if (prediction) {
                    result.setText("Valid");
                } else {
                    result.setText("Forged");
                }

            } catch (IOException ioException) {
                ioException.printStackTrace();
            }

        });

        window.getContentPane().add(BorderLayout.NORTH, menu);
        window.getContentPane().add(BorderLayout.CENTER, imagePanel);
        window.getContentPane().add(BorderLayout.SOUTH, bottomPanel);

        window.setVisible(true);

    }

    public static boolean predict(File input) throws IOException {

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("C:/Users/skymind/Desktop/GP/signature-verification/src/main/java/application/model.zip");

        ImageLoader imageLoader = new ImageLoader(width, height, 3);
        INDArray image = imageLoader.asMatrix(input);

        INDArray result = model.output(image);

        if (result.getDouble(0) < result.getDouble(1)) {
            return true;
        } else {
            return false;
        }

    }

    // Train
    public static InputSplit[] dataLoading() throws IOException {

        // Data Reading
        File file = new ClassPathResource("sign_data").getFile();
        FileSplit fileSplit = new FileSplit(file);

        PathFilter pathFilter = new BalancedPathFilter(rand, allowedFormats, labelMaker);

        InputSplit[] inputSplit = fileSplit.sample(pathFilter, 0.8, 0.2);

        return inputSplit;

    }

    public static void train() throws IOException {

        InputSplit[] data = dataLoading();

        InputSplit trainSplit = data[0];
        InputSplit testSplit = data[1];

        ImageRecordReader trainReader = new ImageRecordReader(height, width, 3, labelMaker);
        ImageRecordReader testReader = new ImageRecordReader(height, width, 3, labelMaker);


        ImageTransform flip = new FlipImageTransform(0);

        List<Pair<ImageTransform, Double>> transforms = Arrays.asList(
                new Pair<>(flip, .5)
        );

        ImageTransform pipeline = new PipelineImageTransform(transforms, true);

        trainReader.initialize(trainSplit, pipeline);
        testReader.initialize(testSplit);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainReader, batchSize, 1, 2);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testReader, batchSize, 1, 2);

        DataNormalization normalization = new ImagePreProcessingScaler();

        normalization.fit(trainIter);
        trainIter.setPreProcessor(normalization);
        testIter.setPreProcessor(normalization);

        MultiLayerNetwork model = new MultiLayerNetwork(cnn());

        model.init();

        model.setListeners(
                new ScoreIterationListener(1)
        );

        model.fit(trainIter, epoch);

        Evaluation evalTrain = model.evaluate(trainIter);
        Evaluation evalTest = model.evaluate(testIter);
        System.out.println("Train Data: " + evalTrain.stats());
        System.out.println("Test Data : " + evalTest.stats());

        ModelSerializer.writeModel(model, "C:/Users/skymind/Desktop/GP/signature-verification/src/main/java/application/model.zip", true);

    }

    public static MultiLayerConfiguration cnn() {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .activation(Activation.RELU)
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder()
                        .nIn(3)
                        .nOut(50)
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .build())
                .layer(new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(new ConvolutionLayer.Builder()
                        .nOut(100)
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .build())
                .layer(new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(200)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nOut(2)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, 3))
                .build();

        return config;
    }

}

class ImagePanel extends JPanel {

    private BufferedImage image;
    private File imgFile;

    public void readImage(File file) throws IOException {
        imgFile = file;
        image = ImageIO.read(file);
    }

    public File getImage() {
        return imgFile;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        g.drawImage(image, 0, 0, this);
    }
}
