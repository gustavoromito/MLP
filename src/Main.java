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

    /** Parametros da Rede Neural */
    public static double TAXA_APRENDIZADO = 0.5;
    public static int NUMERO_EPOCAS = 10000;
    public static int NUMERO_NEURONIOS_CAMADA_SAIDA = 3;

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static double[] insertBiasToHog(float[] input) {
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

    public static void main(String[] args) {

        /* We need to insert bias equals 1 here in first element */

        // Valores da camada de entrada => [b1, x1, x2, x3]

        // Pesos da camada de entrada
        //
        //	| v01, v11, v21, v31 |
        //	| v02, v12, v22, v32 |
        //	| v03, v13, v23, v33 |
        //
//        double[][] pcamadaEntrada = { {2, 2, 2, 2}, {-4, -4, -4, -4}, {6, 6, 6, 6} };
//        double[][] pcamadaEscondida = { {3,3,3,3}, {5,5,5,5}, {7,7,7,7} };

        String[] images = new String[10];
        for (int i = 0; i < images.length; i++) {
            images[i] = "train_53_0100" + i + ".png";
        }

        /* Camada de Saída possui index 0 com bias fixo. Consumir a partir do index 1. */
        for (int i = 0; i < images.length; i++) {
            System.out.println("RODANDO IMAGEM: " + images[i]);

            float[] entrada = hogDescriptor(images[i]);

            int[] valoresEsperados = valoresEsperadosForFileName(images[i]);
            double[] camadaEntrada = insertBiasToHog(entrada);
            double[][] pcamadaEntrada = randomPesos(camadaEntrada);

            double[] camadaEscondida = calcularSomatorio(camadaEntrada, pcamadaEntrada, false);
            double[][] pcamadaEscondida = randomPesos(camadaEscondida);

            double[] camadaSaida = new double[NUMERO_NEURONIOS_CAMADA_SAIDA];

            int j = 0;
            while (TAXA_APRENDIZADO < 1 || j < NUMERO_EPOCAS) {

                //FEED FORWARD
                camadaEscondida = calcularSomatorio(camadaEntrada, pcamadaEntrada, false);
                camadaSaida = calcularSomatorio(camadaEscondida, pcamadaEscondida, true);
                // END FEED FORWARD

                //BACK PROPAGATION
                double[] erroSaida = calcularErro(valoresEsperados, camadaSaida);
                double[][] deltaPesosCamadaEscondida = deltaPesos(erroSaida, camadaEscondida, camadaSaida.length);
                double[] deltaBiasCamadaEscondida = deltaPesosBias(erroSaida);

                double[] erroEntrada = calcularErroFinal(erroSaida, camadaEscondida, pcamadaEscondida);
                double[][] deltaPesosCamadaEntrada = deltaPesos(erroEntrada, camadaEntrada, camadaEscondida.length);
                double[] deltaBiasCamadaEntrada = deltaPesosBias(erroEntrada);

                double[][] novosPesosCamadaEscondida = atualizaPesos(pcamadaEscondida, deltaPesosCamadaEscondida, deltaBiasCamadaEscondida);
                double[][] novosPesosCamadaEntrada = atualizaPesos(pcamadaEntrada, deltaPesosCamadaEntrada, deltaBiasCamadaEntrada);

                pcamadaEntrada = novosPesosCamadaEntrada;
                pcamadaEscondida = novosPesosCamadaEscondida;
                // END BACK PROPAGATION

                TAXA_APRENDIZADO += 0.001;
                j++;
            }

            System.out.println("TERMINOU IMAGEM: " + images[i]);
        }
        System.out.print(true);
    }

    public static int[] valoresEsperadosForFileName(String filename) {
        /*
           train_5a_ = Z
           train_53_ = S
           train_58_ = X
        */
        if (filename.contains("train_5a_")) {
            return new int[]{ 1 , 0 , 0 };
        } else if (filename.contains("train_53_")) {
            return new int[]{ 0 , 1 , 0 };
        } else {
            return new int[]{ 0 , 0 , 1 };
        }
    }

    public static double[] calcularErro(int[] valorEsperado, double[] camadaSaida) {
        double[] erro = new double[camadaSaida.length];
        for(int k = 0; k < camadaSaida.length; k++){
            erro[k] = (valorEsperado[k] - camadaSaida[k]) * derivadaFuncaoExponencial(camadaSaida[k]);
        }
        return erro;
    }
    
    public static double[] calcularErroFinal(double[] erro, double[] camadaEscondida, double[][] pcamadaSaida) {
        double[] aux = new double[pcamadaSaida.length];
        // Parte da somatória para achar os valores de entrada da camada escondida
        for(int i = 0; i < pcamadaSaida.length; i++) {
            double somatoria = 0;
            for(int j = 0; j < erro.length; j++) {
                somatoria += erro[j] * pcamadaSaida[i][j];
            }
            aux[i] = somatoria;
        }

        double[] erroFinal = new double[pcamadaSaida.length];
        for(int k = 0; k < erroFinal.length; k++){
            erroFinal[k] = aux[k] * derivadaFuncaoExponencial(camadaEscondida[k]);
        }

        return erroFinal;
    }
    
    // CAMADAX DEVE SER OU A CAMADA ESCONDIDA OU A CAMADA DE ENTRADA,
    // DEPENDE DE QUAL DELTA ESTAMOS QUERENDO, OU EH O DELTA W OU
    // O DELTA V
    public static double[][] deltaPesos(double[] erro, double[] camadaX, int nextCamadaLength){
        double[][] deltapesos = new double[camadaX.length][nextCamadaLength];
        for(int i = 0; i < deltapesos.length; i++) {
            for(int j = 1; j < erro.length; j++){
                deltapesos[i][j] = TAXA_APRENDIZADO * erro[j] * camadaX[i];
            }
        }
        return deltapesos;
    }
    
    // ARRAY ERRO PODE SER O ERRO DA CAMADA ESCONDIDA OU
    // DA CAMADA DE ENTRADA
    public static double[] deltaPesosBias(double[] erro){
        double[] dpbias = new double[erro.length];
        for(int i = 0; i < dpbias.length; i++){
            dpbias[i] = TAXA_APRENDIZADO * erro[i];
        }
        return dpbias;
    }
    
    public static double[][] atualizaPesos(double[][] pcamadaX, double[][] deltapeso, double[] dpbias){
        double[][] novosPesos = new double[pcamadaX.length][pcamadaX[0].length];
        for(int j = 0; j < pcamadaX.length; j++){
            for(int k = 1; k < deltapeso[j].length; k++){
                novosPesos[j][k] = pcamadaX[j][k] + deltapeso[j][k];
            }
        }
        
        for(int i = 0; i < dpbias.length; i++){
            novosPesos[i][0] = pcamadaX[i][0] + dpbias[i];
        }

        return novosPesos;
    }

    public static double[][] randomPesos(double[] camada) {
        Random r = new Random();

        double[][] pesos = new double[camada.length - 1][camada.length];

        for (int i = 0; i < pesos.length; i++) {
            for (int j = 0; j < pesos[i].length; j++) {
                pesos[i][j] = r.nextDouble();
            }
        }
        return pesos;
    }

    // Retorna um array com os valores dos somatorios para a camada seguinte
    public static double[] calcularSomatorio(double[] camadaX, double[][] camadaPesos, boolean isCamadaSaida) {
        // Variavel para guardar o valor da somatoria da entrada do neuronio * peso do neuronio

        int length = isCamadaSaida ? NUMERO_NEURONIOS_CAMADA_SAIDA : camadaX.length;
        double[] novaCamada = new double[length];

        // Setando o primeiro valor no vetor, que será o bias
        if (!isCamadaSaida) novaCamada[0] = 1;

        // Parte da somatória para achar os valores de entrada da camada escondida
        int initial = (isCamadaSaida) ? 0 : 1;
        for(int i = initial; i < novaCamada.length; i++) {
            double somatoria = 0;
            int index = Math.max(i - 1, 0);
            for(int j = 0; j < camadaPesos[index].length; j++) {
                somatoria += camadaX[j] * camadaPesos[index][j];
            }
            novaCamada[i] = funcaoExponencial(somatoria);
        }

        return novaCamada;
    }


    /* FUNCAO DE ATIVACAO */
    public static double funcaoExponencial(double sum) {
        return 1.0 / (1.0 + Math.exp(-sum));
    }

    // Fx = Resultado da Funcao Exponencial
    public static double derivadaFuncaoExponencial(double fx) {
        return fx * (1.0 - fx);
    }

    public static float[] hogDescriptor(String filename) {
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

                new Size(64, 64), //winSize
                new Size(64, 64), //blocksize
                new Size(64, 64), //blockStride
                new Size(64, 64), //cellSize
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

}
