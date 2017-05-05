import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by gustavoromito on 5/2/17.
 */
public class KFold {

    public static final int NUMBER_OF_FOLDS = 5;
    private static final int NUMBER_OF_IMAGES = 3000;
    private static final int NUMBER_OF_IMAGES_FOR_LETTER = 1000;
    private static final int TAMANHO_AMOSTRA = (int)(0.3 * NUMBER_OF_IMAGES);
    private static final int TAMANHO_AMOSTRA_CADA_LETRA = TAMANHO_AMOSTRA / 3;

    private ArrayList<String> zImagesNames;
    private ArrayList<String> sImagesNames;
    private ArrayList<String> xImagesNames;

    //FOLDS
    private ArrayList<String> fold1;
    private ArrayList<String> fold2;
    private ArrayList<String> fold3;
    private ArrayList<String> fold4;
    private ArrayList<String> fold5;

    public List<ArrayList<String>> folds;

    public KFold() {
        zImagesNames = randomImagesWithPreffix(ProjectHelper.Z_IMAGE_PREFFIX);
        sImagesNames = randomImagesWithPreffix(ProjectHelper.S_IMAGE_PREFFIX);
        xImagesNames = randomImagesWithPreffix(ProjectHelper.X_IMAGE_PREFFIX);

        fold1 = initFold();
        fold2 = initFold();
        fold3 = initFold();
        fold4 = initFold();
        fold5 = initFold();
        folds = new ArrayList<>();

        folds.add(fold1);
        folds.add(fold2);
        folds.add(fold3);
        folds.add(fold4);
        folds.add(fold5);
    }

    public List<ArrayList<String>> getFolds() {
        return folds;
    }

    /**
     * This method is responsible for getting random images names for each letter
     */
    private ArrayList<String> randomImagesWithPreffix(String preffix) {

        ArrayList<String> finalList = new ArrayList<>();
        ArrayList<Integer> alreadyUsedNumbers = new ArrayList<>();

        /** FinalList must hava size equal to Tamanho da Amosta */
        while (finalList.size() < TAMANHO_AMOSTRA_CADA_LETRA) {
            int random = ProjectHelper.randomInt(0, NUMBER_OF_IMAGES_FOR_LETTER - 1);

            /** Imagem já adicionada, pular para a próxima */
            if (alreadyUsedNumbers.contains(random))
                continue;

            String imageName = preffix + ProjectHelper.paddingLeft(random) + ".png";
            finalList.add(imageName);
        }

        return finalList;
    }

    /**
     * This method is responsible for init each fold. It gets 20 random images
     * from each Set: zImagesNames, sImagesNames and xImagesNames
     */
    private ArrayList<String> initFold() {

        ArrayList<String> fold = new ArrayList<>();

        // GET 20 Images for each letter
        fold.addAll(randomElementsFromSet(zImagesNames));
        fold.addAll(randomElementsFromSet(sImagesNames));
        fold.addAll(randomElementsFromSet(xImagesNames));

        return fold;
    }

    /**
     * This method gets 20 images random for the passed set
     */
    private ArrayList<String> randomElementsFromSet(List<String> set) {
        int mustHaveLength = TAMANHO_AMOSTRA_CADA_LETRA / NUMBER_OF_FOLDS;

        ArrayList<String> elements = new ArrayList<>();

        while(elements.size() < mustHaveLength) {

            int randomIndex = ProjectHelper.randomInt(0, set.size() - 1);

            // Removing element from set avoids getting duplicated itens in any fold
            elements.add(set.remove(randomIndex));
        }

        return elements;
    }

    /**
     * Method responsible for validating our MLP
     */

    public void validateMLP() {
        for(int i = 0; i < KFold.NUMBER_OF_FOLDS; i++) {
            List<ArrayList<String>> foldsCopy = this.getFolds();

            List<String> staticFold = foldsCopy.remove(i);

            MLP rede = new MLP(1);

            for(int j = 0; j < foldsCopy.size(); j++) {
                ArrayList<String> fold = foldsCopy.get(j);

                for(int y = 0; y < fold.size(); y++) {

                    String imageName = fold.get(y);
                    int[] esperados = ProjectHelper.valoresEsperadosForFileName(imageName);
                    double[] entrada = ProjectHelper.readImage(imageName, "treinamento");

                    rede.addValorEsperado(esperados);
                    rede.addEntrada(entrada);
                }

            }

            rede.learn();

            ArrayList<double[]> entradas = new ArrayList<>();
            ArrayList<int[]> esperados = new ArrayList<>();
            for(int y = 0; y < staticFold.size(); y++) {
                String imageName = staticFold.get(y);
                int[] valoresEsperados = ProjectHelper.valoresEsperadosForFileName(imageName);
                double[] entrada = ProjectHelper.readImage(imageName, "treinamento");
                entradas.add(entrada);
                esperados.add(valoresEsperados);
            }

            double[] erros = rede.validate(entradas, esperados);
            System.out.println("Erros: " + erros.toString());
        }
    }
}
