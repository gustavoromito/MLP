import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.objdetect.HOGDescriptor;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.List;

/**
 * Created by gustavoromito on 5/1/17.
 */
public class ProjectHelper {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    /**
     * Read Image From File
     */
    public static double[] readImage(String imageName) {
        float[] entrada = hogDescriptor(imageName);
        return insertBiasToHog(entrada);
    }

    /**
     * train_5a_ = Z
     * train_53_ = S
     * train_58_ = X
     */
    public static int[] valoresEsperadosForFileName(String filename) {
        if (filename.contains("train_5a_")) {
            return new int[]{ 1 , 0 , 0 };
        } else if (filename.contains("train_53_")) {
            return new int[]{ 0 , 1 , 0 };
        } else {
            return new int[]{ 0 , 0 , 1 };
        }
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

    /**
     * Responsible for read and return data from Image
     * Using Hog Descriptor
     */
    private static float[] hogDescriptor(String filename) {
//        carrega uma img, o parametro é o caminho para a imagem
        String projectAbsolutePath = "/Users/gustavoromito/Companies/USP Faculdade/IA/dataset1/testes/";
        Mat img = Highgui.imread(projectAbsolutePath + filename);

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

                new Size(16, 16), //winSize
                new Size(16, 16), //blocksize
                new Size(16, 16), //blockStride
                new Size(16, 16), //cellSize
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
    public static void recordConfig() {
        List<String> lines = new ArrayList<>();
        Path file = Paths.get("config.txt");

        String timeStamp = new SimpleDateFormat("dd/MM/yyyy HH:mm").format(Calendar.getInstance().getTime());
        lines.add("Execucao em " + timeStamp);

        try {
            Files.write(file, lines, Charset.forName("UTF-8"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
