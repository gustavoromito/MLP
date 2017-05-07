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
        testMLP();
        testKFold();
    }

    public static void testKFold() {
        System.out.println("--------- RODANDO KFOLD --------");
        KFold kFold = new KFold();
        kFold.validateMLP(ProjectHelper.HOG_EXTRACTOR);
        System.out.println("--------- TÉRMINO DO KFOLD --------");
    }

    public static void testMLP() {
        System.out.println("--------- RODANDO TREINAMENTO DA REDE NEURAL --------");

        List<String> images = ProjectHelper.sourceImages();

        MLP rede = new MLP();

        /* Camada de Saída possui index 0 com bias fixo. Consumir a partir do index 1. */
        for (int i = 0; i < images.size(); i++) {
            String image = images.get(i);
            System.out.println("PROCESSANDO IMAGEM: " + image);

            int[] esperados = ProjectHelper.valoresEsperadosForFileName(image);
            double[] entrada = ProjectHelper.readImage(images.get(i), null, ProjectHelper.HOG_EXTRACTOR);

            rede.addValorEsperado(esperados);
            rede.addEntrada(entrada);

            System.out.println("TERMINOU PROCESSAMENTO IMAGEM: " + image);
        }

        double[] learningErrors = rede.learn();
        List<KFold.AttemptError> errors = new ArrayList<>();
        for(int i = 0; i < learningErrors.length; i++) {
            errors.add(new KFold.AttemptError(i, learningErrors[i], 999));
        }
        ProjectHelper.recordErrorTxt(errors, "errorsTestes.txt");
        ProjectHelper.recordConfig(rede);
        ProjectHelper.recordModel(rede);

        System.out.println("--------- FIM DO TREINAMENTO DA REDE NEURAL --------");
    }


}
