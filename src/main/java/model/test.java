package model;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.PathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class test {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(test.class);

    private static int seed = 123;
    private static final Random randNumGen = new Random(seed);
    static String[] allowedFormats = BaseImageLoader.ALLOWED_FORMATS;
    static PathLabelGenerator labelGenerator = new ParentPathLabelGenerator();
    static double learningRate = 0.001;
    static int nEpoch = 2;
    static double splitRatio = 0.8;

    static int height = 224;
    static int width = 224;
    static int nChannel = 1;
    private static int batchSize = 8;
    private static int numClasses = 2;

    public static void main(String[] args) throws Exception {

        File inputFile = new ClassPathResource("sign_data/train/").getFile();
        FileSplit fileSplit = new FileSplit(inputFile);

        PathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedFormats, labelGenerator);

        InputSplit[] sample = fileSplit.sample(pathFilter, splitRatio, 1 - splitRatio);
        InputSplit trainData = sample[0];
        InputSplit testData = sample[1];

        ImageTransform hFlip = new FlipImageTransform(1);
        ImageTransform rotate = new RotateImageTransform(15);
        ImageTransform showImage = new ShowImageTransform("Transformed Images", 1000);

        List<Pair<ImageTransform, Double>> transform = Arrays.asList(
                new Pair<>(hFlip, 0.5),
                new Pair<>(rotate, 0.3),
                new Pair<>(showImage, 0.1)
        );

        ImageTransform pipeline = new PipelineImageTransform(transform, false);

        ImageRecordReader trainRR = new ImageRecordReader(height, width, nChannel, labelGenerator);
        trainRR.initialize(trainData, pipeline);
        ImageRecordReader testRR = new ImageRecordReader(height, width, nChannel, labelGenerator);
        testRR.initialize(testData);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, numClasses); // labelIndex for Image data always 1
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, numClasses);

        //load vgg16 zoo model
        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
        System.out.println(vgg16.summary());

        // Override the setting for all layers that are not "frozen".
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .updater(new Nesterovs(5e-4, 0.9))
                .seed(seed)
                .build();

        //Construct a new model with the intended architecture and print summary
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("fc2") //the specified layer and below are "frozen"
                .nInReplace("block1_conv1", nChannel, WeightInit.XAVIER)
                .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                .nIn(4096).nOut(numClasses)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX).build(),
                        "fc2")
                .build();
        log.info(vgg16Transfer.summary());

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        vgg16Transfer.setListeners(
                new StatsListener(statsStorage),
                new ScoreIterationListener(5),
                new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END),
                new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END)
        );

        vgg16Transfer.fit(trainIter, nEpoch);

        System.out.println(vgg16Transfer.evaluate(trainIter).stats());
        System.out.println(vgg16Transfer.evaluate(testIter).stats());
    }
}
