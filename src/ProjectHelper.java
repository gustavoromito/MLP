import org.opencv.core.*;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.highgui.Highgui;
import org.opencv.objdetect.HOGDescriptor;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Created by gustavoromito on 5/1/17.
 */
public class ProjectHelper {

    public static final String Z_IMAGE_PREFFIX = "train_5a_";
    public static final String S_IMAGE_PREFFIX = "train_53_";
    public static final String X_IMAGE_PREFFIX = "train_58_";

    public static final int mWinSize = 32;
    public static final int mBlockSize = 32;
    public static final int mBlockStride = 32;
    public static final int mCellSize = 32;

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static final int HOG_EXTRACTOR = 0;
    public static final int SIFT_EXTRACTOR = 1;

    private static final int MAX_KEYPOINTS = 4;

    public static final int SIFT_ENTRY_SIZE = MAX_KEYPOINTS * 128 + 1;
    public static final int HOG_ENTRY_SIZE = 145;

    /**
     * Random number between two integers
     */

    public static int randomInt(int lowerBound, int upperBound) {
        Random r = new Random();
        return r.nextInt((upperBound - lowerBound) + 1) + lowerBound;
    }

    /**
     * Converts number from '1' to '00001'
     */
    public static String paddingLeft(int number, int size) {
        return String.format("%0"+ size + "d", number);
    }
    /**
     * Read Image From File
     */
    public static double[] readImage(String imageName, String folderName, int extractor) {
        float[] entrada;

        Mat img = getImageNamed(imageName, folderName);

        if (extractor == SIFT_EXTRACTOR)
            entrada = SIFTExtractor(img);
        else
            entrada = hogDescriptor(img);
        return insertBiasToHog(entrada);
    }

    public static int[] valoresEsperadosForFileName(String filename) {
        String[] letters = Letters.getAllPreffix();
        int[] valoresEsperados = new int[letters.length]; // 26 letters

        for (int i = 0; i < letters.length; i++) {
            String preffix = letters[i];
            if (filename.contains(preffix)) {
                valoresEsperados[i] = 1;
                break;
            }            
        }

        return valoresEsperados;
    }

    /**
     * We need to insert Bias to the first Element
     * Of Hog Descriptor
     */
    private static double[] insertBiasToHog(float[] input) {
        if (input == null) {
            return null; // Or throw an exception - your choice
        }
        double[] output = new double[input.length + 1];
        /* First element is the Biar */
        output[0] = 1;

        for (int i = 1; i <= input.length; i++) {
            output[i] = input[i - 1];
        }
        return output;
    }

    private static Mat getImageNamed(String filename, String folderName) {
        String projectAbsolutePath = "/Users/gustavoromito/Companies/USP Faculdade/IA/dataset1/";
        projectAbsolutePath += (folderName == null) ? "testes/" : (folderName + "/");

        Mat img = Highgui.imread(projectAbsolutePath + filename);
        return img;
    }

    /**
     * Responsible for read and return data from Image
     * Using SIFT Descriptor
     */
    private static float[] SIFTExtractor(Mat img) {

        FeatureDetector detector = FeatureDetector.create(FeatureDetector.SIFT);

        MatOfKeyPoint keyPoints = new MatOfKeyPoint();
        detector.detect(img, keyPoints);

        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.SIFT);
        MatOfDMatch descriptors = new MatOfDMatch();
        extractor.compute(img, keyPoints, descriptors);
        descriptors.channels();
        descriptors.depth();

        int rows = Math.min(MAX_KEYPOINTS, descriptors.rows());
        int cols = descriptors.cols();
        float[] a = new float[rows * cols];
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                a[i * j] = (float)descriptors.get(i, j)[0];
            }
        }

        return a;
    }

    /**
     * Responsible for read and return data from Image
     * Using Hog Descriptor
     */
    private static float[] hogDescriptor(Mat img) {
        //HOG
        HOGDescriptor hog = new HOGDescriptor(
                //winSize: size of the digit images in our dataset

                //blocksize: the notion of blocks exist to tackle illumination variation.
                //A large block size makes local changes less significant while a smaller
                //block size weights local changes more. Typically blockSize is set to 2 x cellSize

                //blockStride: The blockStride determines the overlap between neighboring blocks and controls the
                //degree of contrast normalization. Typically a blockStride is set to 50% of blockSize.

                //cellSize: image is represented by 128×128 = 16384 numbers.The size of descriptor typically is much smaller than
                //the number of pixels in an image. The cellSize is chosen based on the scale of the features important
                //to do the classification. A very small cellSize would blow up the size of the
                //feature vector and a very large one may not capture relevant information. You should test this yourself.

                //nbins: sets the number of bins in the histogram of gradients. The authors of the HOG paper had
                //recommended a value of 9 to capture gradients between 0 and 180 degrees in 20 degrees increments.

                new Size(mWinSize, mWinSize), //winSize
                new Size(mBlockSize, mBlockSize), //blocksize
                new Size(mBlockStride, mBlockStride), //blockStride
                new Size(mCellSize, mCellSize), //cellSize
                9 //nbins
        );

        //armazena os valores do hog
        MatOfFloat floats = new MatOfFloat();

        //calcula o hog
        hog.compute(img, floats);

        floats.channels();
        floats.depth();
        float a[] = floats.toArray();

        return a;
    }

    /**
     * Config.txt
     */
    public static void recordConfig(MLP rede, int extractorType) {
        List<String> lines = new ArrayList<>();
        Path file = Paths.get("config.txt");

        String timeStamp = new SimpleDateFormat("dd/MM/yyyy HH:mm").format(Calendar.getInstance().getTime());
        lines.add("Execucao em " + timeStamp + "\n");

        lines.add("extrator : " + ((extractorType == SIFT_EXTRACTOR) ? "SIFT" : "HOG"));
        lines.add("extrator_orientacoes : 9");
        lines.add("extrator_pixel_por_celula : " + mWinSize / mCellSize);
        lines.add("extrator_celula_por_bloco : " + mWinSize / mBlockSize);

        lines.add("rede_alpha_inicial : " + rede.mTaxaAprendizadoInicial);
        lines.add("rede_alpha_final : " + rede.getTaxaAprendizado());
        lines.add("rede_camada_1_neuronios : " + rede.getCamadaEscondida().length);
        lines.add("rede_camada_1_funcao_de_ativacao  : " + "sigmoide");
        lines.add("rede_camada_2_neuronios : " + rede.getCamadaSaida().length);
        lines.add("rede_camada_2_funcao_de_ativacao  : " + "sigmoide");
        lines.add("rede_inicializacao_pesos  : " + "aleatoria");
        lines.add("rede_min_epocas  : " + rede.mNumeroMinEpocas);
        lines.add("rede_max_epocas  : " + rede.getNumeroEpocas());
        lines.add("rede_parada_antecipada : " + rede.earlier_stopped);

        try {
            Files.write(file, lines, Charset.forName("UTF-8"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Error.txt
     */
    public static void recordErrorTxt(List<KFold.AttemptError> errors, String filename) {
        List<String> lines = new ArrayList<>();
        Path file = Paths.get(filename == null ? "error.txt" : filename);

        String timeStamp = new SimpleDateFormat("dd/MM/yyyy HH:mm").format(Calendar.getInstance().getTime());
        lines.add("Execucao em " + timeStamp + "\n");

        for (KFold.AttemptError error : errors) {
            lines.add(error.getEpoca() + ";" + error.getErroTreinamento() + ";" + error.getErroValidacao());
        }

        try {
            Files.write(file, lines, Charset.forName("UTF-8"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Model.txt
     */
    public static void recordModel(MLP rede) {
        List<String> lines = new ArrayList<>();
        Path file = Paths.get("model.txt");

        String timeStamp = new SimpleDateFormat("dd/MM/yyyy HH:mm").format(Calendar.getInstance().getTime());
        lines.add("Execucao em " + timeStamp + "\n");

        lines.add(serializeMatrix(rede.getPesosCamadaEntrada()));
        lines.add("\n");
        lines.add(serializeMatrix(rede.getPesosCamadaEscondida()));

        try {
            Files.write(file, lines, Charset.forName("UTF-8"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static String serializeMatrix(double[][] matrix) {

        String line = "[";
        for(int i = 0; i < matrix.length; i++) {
            line += Arrays.toString(matrix[i]);

            if (i != matrix.length - 1)
                line += ", ";
        }

        line += "]";
        return line;

    }

    public static List<String> sourceImages() {
        ArrayList<String> images = new ArrayList<>();
        images.addAll(Arrays.asList(imagesForPreffix(Z_IMAGE_PREFFIX)));
        images.addAll(Arrays.asList(imagesForPreffix(S_IMAGE_PREFFIX)));
        images.addAll(Arrays.asList(imagesForPreffix(X_IMAGE_PREFFIX)));

        return images;
    }

    private static String[] imagesForPreffix(String preffix) {
        String[] images = new String[300];
        for (int i = 0; i < images.length; i++) {
            String id = paddingLeft(i, 3);
            images[i] = preffix + "01" + id + ".png";
        }

        return images;
    }


}
