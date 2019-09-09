import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Random;
/**
 * @author salman khan, 
 */
public class pseSNC_DNC_TNC_L2 {
    private static Logger log = LoggerFactory.getLogger(pseSNC_DNC_TNC_L2.class);
    public static void main(String[] args) throws  Exception {
		
       PrintWriter pr = new PrintWriter("accuracy.txt");
	   long lr[] ={0.09,0.08,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9};
	   int numLinesToSkip = 0;
       char delimiter = ',';
       int labelIndex = 86, numClasses = 2, batchSize = 1418;
       int Inputs = 75, outputs = 2;
       double acc[] = new double[length];
      
	   RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
       recordReader.initialize(new FileSplit(new ClassPathResource("pse_SDT_L2.txt").getFile()));
       DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
       DataSet alldataset = iterator.next();
       DataNormalization normalizer = new NormalizerStandardize();
       normalizer.fit(alldataset);           //Collect the statistics mean or stdev from the training data.
       normalizer.transform(alldataset);     //Apply normalization on all dataset i.e. Z-score
	   allData.shuffle();      
       
	   int k =10;							// k-fold value     
       KFoldIterator kf = new KFoldIterator(k, allData);
        for (int i = 0; i < k; i++) {
            DataSet trainingData = kf.next();
            DataSet testData = kf.testFold();

			MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.seed(6)
			.weightInit(WeightInit.XAVIER)
			.updater(Updater.ADAGRAD)
			.activation(Activation.TANH)					//activation function Tanh 
			.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
			.updater(new Nesterovs(0.9))   //momentum 0.9
			.updater(new Sgd(0.1))			// learning rate
			.l2(1e-4)
			.list()

			.layer(0, new DenseLayer.Builder().nIn(Inputs).nOut(74).build())
			.layer(1, new DenseLayer.Builder().nIn(74).nOut(64).build())
			.layer(2, new DenseLayer.Builder().nIn(64).nOut(36).build())
			.layer(3, new DenseLayer.Builder().nIn(36).nOut(23).activation(Activation.SIGMOID).build())
			.layer(4, new DenseLayer.Builder().nIn(23).nOut(19).build())
			.layer(5, new DenseLayer.Builder().nIn(19).nOut(3).build())
			.layer(6, new OutputLayer.Builder().nIn(3).nOut(outputs)
				.lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
				.activation(Activation.SOFTMAX)
				.build())
				.pretrain(false).backprop(true)
				.build();

			MultiLayerNetwork model = new MultiLayerNetwork(conf);
			model.init();
			model.setListeners(new ScoreIterationListener(100));
	
	
			UIServer uiServer = UIServer.getInstance();
			StatsStorage statsStorage = new InMemoryStatsStorage();
			int listenerFrequency = 1;
			model.setListeners(new StatsListener(statsStorage, listenerFrequency));
			uiServer.attach(statsStorage);

			for (int j = 0; j < 500; j++) 
				model.fit(trainingData);

			//evaluate the model on the test set
			Evaluation eval = new Evaluation(2);
			INDArray output = model.output(testData.getFeatures());
			eval.eval(testData.getLabels(), output);
			System.out.println(eval.stats());
			acc[i] = eval.accuracy();
    
 
		   //ROC, work on binary class
			File file = new File("abc.html");
			ROC roc = new ROC();
			roc.eval(testData.getLabels(), output);
			EvaluationTools.exportRocChartsToHtmlFile(roc,file);
			log.info(roc.stats());

		   //Regression , R^2, MSE, etc....
			RegressionEvaluation eval1 = new RegressionEvaluation(2);    //correct
			eval1.eval(testData.getLabels(), output);
			log.info(eval1.stats());
		}
        for (int i=0; i<k ; i++)
          pr.println(acc[i]); 
        pr.println("-----------------------------------");
        double sum = Arrays.stream(acc).sum();
        pr.println("K-Fold Accuracy = " + sum /k );
        pr.println("-----------------------------------");
        pr.close();
    }
}

