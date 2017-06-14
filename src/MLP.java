import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by gustavoromito on 5/1/17.
 */
public class MLP {

    private static final int NUMERO_NEURONIOS_CAMADA_SAIDA = Letters.getAllPreffix().length;

    public double mTaxaAprendizadoInicial;
    private double mTaxaAprendizado;
    public int mNumeroMinEpocas = 200;
    private int mNumeroEpocas;

    public String earlier_stopped = "nenhuma";

    public int STOP_WINDOWS = 5;

    private FuncaoAtivacao mFin;
    private int[] mValoresEsperados;

    private List<double[]> mEntradas = new ArrayList<>();
    private List<int[]> mEntradasValoresEsperados = new ArrayList<>();

    // Camada Entrada
    private double[] mCamadaEntrada;
    private double[][] mPCamadaEntrada;

    // Camada Escondida
    private double[] mCamadaEscondida;

    private double[][] mPCamadaEscondida;

    private double[] mCamadaSaida = new double[NUMERO_NEURONIOS_CAMADA_SAIDA];

    public MLP() {
        /** Default constructor with Tamanho Entrada */
        this(145);
    }

    // Inicializalicao da Rede Neural
    public MLP(int tamanhoEntrada) {
        this(500, tamanhoEntrada);
    }

    // Inicializalicao da Rede Neural
    public MLP(int numeroEpocas, int tamanhoEntrada) {
        this(numeroEpocas, 0.01, tamanhoEntrada);
    }

    public MLP(int numberEpocas, double taxaAprendizado, int tamanhoEntrada) {
        mFin = new FuncaoAtivacao(FuncaoAtivacao.Type.Sigmoide);

        mCamadaEntrada = new double[tamanhoEntrada];
        mPCamadaEntrada = randomPesos(mCamadaEntrada);

        // Inicializa Camada Escondida
        mCamadaEscondida = new double[tamanhoEntrada];
        mPCamadaEscondida = randomPesos(mCamadaEscondida);

        mNumeroEpocas = numberEpocas;
        mTaxaAprendizado = taxaAprendizado;
        mTaxaAprendizadoInicial = taxaAprendizado;
    }

    public void addEntrada(double[] entrada) {
        mEntradas.add(entrada);
    }

    public void addValorEsperado(int[] valoresEsperados) {
        mEntradasValoresEsperados.add(valoresEsperados);
    }

    private double[][] randomPesos(double[] camada) {
        Random r = new Random();

        double[][] pesos = new double[camada.length - 1][camada.length];

        for (int i = 0; i < pesos.length; i++) {
            for (int j = 0; j < pesos[i].length; j++) {
                pesos[i][j] = r.nextDouble();
            }
        }
        return pesos;
    }

    // Fim da Inicialização

    /**
     * Getters
     */
    public double getTaxaAprendizado() {
        return mTaxaAprendizado;
    }

    public double[] getCamadaEscondida() {
        return mCamadaEscondida;
    }

    public double[] getCamadaSaida() {
        return mCamadaSaida;
    }

    public int getNumeroEpocas() {
        return mNumeroEpocas;
    }

    public double[][] getPesosCamadaEntrada() {
        return mPCamadaEntrada;
    }

    public double[][] getPesosCamadaEscondida() {
        return mPCamadaEscondida;
    }

    /**
     * Feed Forward
     */
    // Retorna um array com os valores dos somatorios para a camada seguinte
    private double[] calcularSomatorio(double[] camadaX, double[][] camadaPesos, boolean isCamadaSaida) {
        // Variavel para guardar o valor da somatoria da entrada do neuronio * peso do neuronio

        int length = isCamadaSaida ? NUMERO_NEURONIOS_CAMADA_SAIDA : camadaX.length;
        double[] novaCamada = new double[length];

        // Setando o primeiro valor no vetor, que será o bias
        if (!isCamadaSaida) novaCamada[0] = 1;

        // Parte da somatória para achar os valores de entrada da camada escondida
        int initial = (isCamadaSaida) ? 0 : 1;
        for(int i = initial; i < novaCamada.length; i++) {
            double somatoria = 0;
            int index = isCamadaSaida ? i : i - 1;
            for(int j = 0; j < camadaPesos[index].length; j++) {
                somatoria += camadaX[j] * camadaPesos[index][j];
            }
            novaCamada[i] = mFin.execute(somatoria);
        }

        return novaCamada;
    }

    /**
     * Back Propagation
     */
    // CAMADAX DEVE SER OU A CAMADA ESCONDIDA OU A CAMADA DE ENTRADA,
    // DEPENDE DE QUAL DELTA ESTAMOS QUERENDO, OU EH O DELTA W OU
    // O DELTA V
    private double[][] deltaPesos(double[] erro, double[] camadaX, int nextCamadaLength){
        double[][] deltapesos = new double[camadaX.length][nextCamadaLength];
        for(int i = 0; i < deltapesos.length; i++) {
            for(int j = 0; j < erro.length; j++){
                deltapesos[i][j] = mTaxaAprendizado * erro[j] * camadaX[i];
            }
        }
        return deltapesos;
    }

    // ARRAY ERRO PODE SER O ERRO DA CAMADA ESCONDIDA OU
    // DA CAMADA DE ENTRADA
    private double[] deltaPesosBias(double[] erro){
        double[] dpbias = new double[erro.length];
        for(int i = 0; i < dpbias.length; i++){
            dpbias[i] = mTaxaAprendizado * erro[i];
        }
        return dpbias;
    }

    private double[] calcularErro(int[] valorEsperado, double[] camadaSaida) {
        double[] erro = new double[camadaSaida.length];
        for(int k = 0; k < camadaSaida.length; k++){
            erro[k] = (valorEsperado[k] - camadaSaida[k]) * mFin.derivate(camadaSaida[k]);
        }
        return erro;
    }

    private double[] calcularErroFinal(double[] erro, double[] camadaEscondida, double[][] pcamadaSaida) {
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
            erroFinal[k] = aux[k] * mFin.derivate(camadaEscondida[k + 1]);
        }

        return erroFinal;
    }

    /**
     * Atualização dos Pesos
     */
    private static double[][] atualizaPesos(double[][] pcamadaX, double[][] deltapeso, double[] dpbias) {
        double[][] novosPesos = new double[pcamadaX.length][pcamadaX[0].length];
        for(int j = 0; j < pcamadaX.length; j++){
            for(int k = 1; k < deltapeso[j].length; k++){
                novosPesos[j][k] = pcamadaX[j][k] + deltapeso[j][k - 1];
            }
        }

        for(int i = 0; i < dpbias.length; i++){
            novosPesos[i][0] = pcamadaX[i][0] + dpbias[i];
        }

        return novosPesos;
    }

    /**
     * Validação da Rede Neural
     */
    public double[] validate(List<double[]> entradas, List<int[]> esperados) {
        mEntradas = entradas;
        mEntradasValoresEsperados = esperados;
        return executeEpocas(false);
    }


    public double[] learn() {
        return executeEpocas(true);
    }

    /**
     * Métodos responsáveis por executar a Rede Neural
     */
    private double[] executeEpocas(boolean isLearning) {
        int j = 0;

        double[] errosEpocas = new double[mNumeroEpocas];
        int increasedErrorCount = 0;
        while (j < mNumeroEpocas) {

            System.out.println("EXECUTANDO ÉPOCA: " + j);
            double erroDaEpoca = executarEpoca(isLearning);
            errosEpocas[j] = erroDaEpoca;

            System.out.println("ERRO DA ÉPOCA " + j + ": " + erroDaEpoca);
            mTaxaAprendizado = Math.min(0.999, mTaxaAprendizado + 1.0 / mNumeroEpocas);

            if (j >= mNumeroMinEpocas) {
                // Validate Inflexao do Erro
                if (errosEpocas[j] > errosEpocas[j - 1]) {
                    increasedErrorCount++;
                    if (increasedErrorCount == STOP_WINDOWS) {
                        earlier_stopped = "deteccao da inflexao da curva de erro de validacao";
                        break;
                    }
                } else {
                    increasedErrorCount = 0;
                }

            }


            j++;
        }

        return errosEpocas;
    }

    private double executarEpoca(boolean shouldLearn) {

        double erroMedio = 0.0;

        for (int i = 0; i < mEntradas.size(); i++) {
            double[] entrada = mEntradas.get(i);
            int[] esperados = mEntradasValoresEsperados.get(i);

            mCamadaEntrada = entrada;
            mValoresEsperados = esperados;

            executarEntrada(shouldLearn);

            double erroDaImagem = 0.0;
            for(int j = 0; j < NUMERO_NEURONIOS_CAMADA_SAIDA; j++) {
                double resultado = mCamadaSaida[j];
                double valorEsperado = mValoresEsperados[j];

                erroDaImagem += Math.pow(valorEsperado - resultado, 2);
            }

            erroDaImagem = erroDaImagem * 0.5;
            erroMedio += erroDaImagem;
        }

        erroMedio = erroMedio / mEntradas.size();
        return erroMedio;

    }

    private void executarEntrada(boolean shouldLearn) {
        stepFeedForward();
        if (shouldLearn)
            stepBackPropagation();
    }

    private void stepFeedForward() {
        //FEED FORWARD
        mCamadaEscondida = calcularSomatorio(mCamadaEntrada, mPCamadaEntrada, false);
        mCamadaSaida = calcularSomatorio(mCamadaEscondida, mPCamadaEscondida, true);
        // END FEED FORWARD
    }

    private void stepBackPropagation() {
        //BACK PROPAGATION
        double[] erroSaida = calcularErro(mValoresEsperados, mCamadaSaida);
        double[][] deltaPesosCamadaEscondida = deltaPesos(erroSaida, mCamadaEscondida, mCamadaSaida.length + 1);
        double[] deltaBiasCamadaEscondida = deltaPesosBias(erroSaida);

        double[] erroEntrada = calcularErroFinal(erroSaida, mCamadaEscondida, mPCamadaEscondida);
        double[][] deltaPesosCamadaEntrada = deltaPesos(erroEntrada, mCamadaEntrada, mCamadaEscondida.length);
        double[] deltaBiasCamadaEntrada = deltaPesosBias(erroEntrada);

        double[][] novosPesosCamadaEscondida = atualizaPesos(mPCamadaEscondida, deltaPesosCamadaEscondida, deltaBiasCamadaEscondida);
        double[][] novosPesosCamadaEntrada = atualizaPesos(mPCamadaEntrada, deltaPesosCamadaEntrada, deltaBiasCamadaEntrada);

        mPCamadaEntrada = novosPesosCamadaEntrada;
        mPCamadaEscondida = novosPesosCamadaEscondida;
        // END BACK PROPAGATION
    }

}
