import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.objdetect.HOGDescriptor;

import java.util.Random;

/**
 * Created by gustavoromito on 4/15/17.
 */
public class Main {

    public static double taxa_aprendizado = 0.01;
    public static int numero_epocas = 10;

    // taxa de aprendizado
    // numero de epocas
    // condição de parada:
    //  - por epoca
    //  - por erro
    //  - taxa de aprendizado
    // Estagio FeedForward
    // Estagio Retropropagaçao do erro
    // Estagio Atualização dos Pesos
    // Bias fixo
    // Matriz Camada de Entrada
    // Matriz de Pesos Entrada para Camada Interna
    // Matriz de Pesos Camada Interna para Saída
    // Uma função para função de ativação
    // uma Camada escondida
    // uma Camada de Saída

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {

        float[] entrada = hogDescriptor();
        /* We need to insert bias equals 1 here in first element */

        int[] camadaEntrada = {1, 4, 6, 8};

        double[][] pesos = randomPesos(camadaEntrada);

        // Valores da camada de entrada => [b1, x1, x2, x3]

        // Pesos da camada de entrada
        //
        //	| v01, v11, v21, v31 |
        //	| v02, v12, v22, v32 |
        //	| v03, v13, v23, v33 |
        //
        int[][] pcamadaEntrada = { {2, 2, 2, 2}, {-4, -4, -4, -4}, {6, 6, 6, 6} };
        int[][] pcamadaEscondida = { {3,3,3,3}, {5,5,5,5}, {7,7,7,7} };

        int[] camadaEscondida = calcularSomatorio(camadaEntrada, pcamadaEntrada);

        /* Camada de Saída possui index 0 com bias fixo. Consumir a partir do index 1. */
        int[] camadaSaida = calcularSomatorio(camadaEscondida, pcamadaEscondida);
    }

    public static int[] calcularErro(int[] camadaEscondida, int valorEsperado, int[] camadaSaida) {
        int[] erro = new int[camadaSaida.length-1];
        for(int k=0; k<camdaSaida.length-1; k++){
            erro[k]=((valorEsperado-camadaEscondida[k])*(camadaSaida[k]*(1-camadaSaida[k])));
        }
        return erro;
    }

//    public static double[][] inicializacaoPesos(int[] fromCamada, int[] toCamada, int fromLayer, int toLayer) {
//        int numberOfNeuronsInputLayer = fromCamada.length;
//        int numberOfNeuronsCurrentLayer = toCamada.length;
//
//        double betaValue = 0.7d * Math.pow(numberOfNeuronsInputLayer, 1d / numberOfNeuronsCurrentLayer);
//
//
//
//    }

    public static double[][] randomPesos(int[] camada) {
        Random r = new Random();

        double[][] pesos = new double[camada.length - 1][camada.length];

        for (int i = 0; i < camada.length - 1; i++) {
            for (int j = 0; j < camada.length; j++) {
                pesos[i][j] = r.nextDouble();
            }
        }
        return pesos;
    }

    // Retorna um array com os valores dos somatorios para a camada seguinte
    public static int[] calcularSomatorio(int[] camadaX, int[][] camadaPesos) {
        // Variavel para guardar o valor da somatoria da entrada do neuronio * peso do neuronio

        int[] novaCamada = new int[4];
        // Setando o primeiro valor no vetor, que será o bias
        novaCamada[0] = 1;
        // Contador que auxiliará para setar os valores no array da camada escondida
        int aux = 1;

        // Parte da somatória para achar os valores de entrada da camada escondida
        for(int i = 0; i < camadaPesos.length; i++) {
            int somatoria = 0;
            for(int j = 0; j < camadaPesos[i].length; j++) {
                somatoria += camadaX[j] * camadaPesos[i][j];
            }
            novaCamada[aux] = funcaoExponencial(somatoria);
            aux++;
        }

        return novaCamada;
    }

    /*
    public static int funcaoAtivacao(int sum) {
        int LIMIAR = 10;
        double fx = funcaoExponencial(sum);

        return (fx >= LIMIAR) ? 1 : -1;
    }*/

    public static double funcaoExponencial(int sum) {
        return 1 / (1 + Math.exp(-sum));
    }

    /*public static double derivadaFuncaoExponencial(int sum) {
        double fx = funcaoAtivacao(sum);

        return fx * (1 - fx);
    }*/

    public static float[] hogDescriptor() {
//        carrega uma img, o parametro é o caminho para a imagem
        String projectAbsolutePath = "/Users/gustavoromito/Companies/USP Faculdade/IA/Projeto/out/production/Projeto/";
        Mat img = Highgui.imread(projectAbsolutePath + "train_5a_00000.png");

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

        for (int i = 0; i < a.length; i++) {
            System.out.print(a[i] + " ");
            if (i % 6 == 1)
                System.out.print("END OF LINE\n");
        }

        return a;
    }

}
