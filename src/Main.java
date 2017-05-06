import java.util.ArrayList;
import java.util.List;

/**
 * Created by gustavoromito on 4/15/17.
 */
public class Main {

    public static void main(String[] args) {
        validateMLP();
    }

    /** K-Fold */
    public static void validateMLP() {
//        testKFold();
        testMLP();
    }

    public static void testKFold() {
        KFold kFold = new KFold();
        kFold.validateMLP();
    }

    public static void testMLP() {
         /* We need to insert bias equals 1 here in first element */

        String[] images = new String[10];
        for (int i = 0; i < images.length; i++) {
            images[i] = "train_58_0100" + i + ".png";
        }

        MLP rede = new MLP();

        /* Camada de Saída possui index 0 com bias fixo. Consumir a partir do index 1. */
        for (int i = 0; i < images.length; i++) {
            System.out.println("PROCESSANDO IMAGEM: " + images[i]);

            int[] esperados = ProjectHelper.valoresEsperadosForFileName(images[i]);
            double[] entrada = ProjectHelper.readImage(images[i], null);

            rede.addValorEsperado(esperados);
            rede.addEntrada(entrada);

            System.out.println("TERMINOU PROCESSAMENTO IMAGEM: " + images[i]);
        }

        double[] learningErrors = rede.learn();
        List<KFold.AttemptError> errors = new ArrayList<>();
        for(int i = 0; i < learningErrors.length; i++) {
            errors.add(new KFold.AttemptError(i, learningErrors[i], 999));
        }
        ProjectHelper.recordErrorTxt(errors, "errorsTestes.txt");
        ProjectHelper.recordConfig(rede);
    }


}
