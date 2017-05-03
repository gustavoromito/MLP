import java.util.ArrayList;
import java.util.List;

/**
 * Created by gustavoromito on 5/2/17.
 */
public class KFold {

    private static final int NUMBER_OF_FOLDS = 5;
    private static final int NUMBER_OF_IMAGES = 3000;
    private static final int TAMANHO_AMOSTRA = (int)(0.3 * NUMBER_OF_IMAGES);
    private static final int TAMANHO_AMOSTRA_CADA_LETRA = TAMANHO_AMOSTRA / 3;

    private List<String> zImagesNames;
    private List<String> sImagesNames;
    private List<String> xImagesNames;

    //FOLDS
    private List<String> fold1;
    private List<String> fold2;
    private List<String> fold3;
    private List<String> fold4;
    private List<String> fold5;

    public KFold() {
        zImagesNames = randomImagesWithPreffix(ProjectHelper.Z_IMAGE_PREFFIX);
        sImagesNames = randomImagesWithPreffix(ProjectHelper.S_IMAGE_PREFFIX);
        xImagesNames = randomImagesWithPreffix(ProjectHelper.X_IMAGE_PREFFIX);

        fold1 = initFold();
        fold2 = initFold();
        fold3 = initFold();
        fold4 = initFold();
        fold5 = initFold();
    }

    /**
     * This method is responsible for getting random images names for each letter
     */
    private ArrayList<String> randomImagesWithPreffix(String preffix) {

        ArrayList<String> finalList = new ArrayList<>();
        ArrayList<Integer> alreadyUsedNumbers = new ArrayList<>();

        /** FinalList must hava size equal to Tamanho da Amosta */
        while (finalList.size() < TAMANHO_AMOSTRA_CADA_LETRA) {
            int random = ProjectHelper.randomInt(0, NUMBER_OF_IMAGES - 1);

            /** Imagem já adicionada, pular para a próxima */
            if (alreadyUsedNumbers.contains(random))
                continue;

            String imageName = preffix + ProjectHelper.paddingLeft(random);
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
}
