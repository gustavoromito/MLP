/**
 * Created by gustavoromito on 4/15/17.
 */
public class Main {

    public static void main(String[] args) {

        /* We need to insert bias equals 1 here in first element */

        String[] images = new String[10];
        for (int i = 0; i < images.length; i++) {
            images[i] = "train_58_0100" + i + ".png";
        }

        /* Camada de Saída possui index 0 com bias fixo. Consumir a partir do index 1. */
        for (int i = 0; i < images.length; i++) {
            System.out.println("RODANDO IMAGEM: " + images[i]);

            int[] esperados = ProjectHelper.valoresEsperadosForFileName(images[i]);
            double[] entrada = ProjectHelper.readImage(images[i]);
            MLP rede = new MLP(entrada, esperados);

            ProjectHelper.recordConfig();
            double[] resultado = rede.executarEpocas();

            System.out.println("TERMINOU IMAGEM: " + images[i]);
        }
    }



}
