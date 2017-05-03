import java.util.Random;

/**
 * Created by gustavoromito on 5/1/17.
 */
public class MLP {

    private static final int NUMERO_NEURONIOS_CAMADA_SAIDA = 3;

    private double mTaxaAprendizado;
    private int mNumeroEpocas;

    private FuncaoAtivacao mFin;
    private int[] mValoresEsperados;

    // Camada Entrada
    private double[] mCamadaEntrada;
    private double[][] mPCamadaEntrada;

    // Camada Escondida
    private double[] mCamadaEscondida;
    private double[][] mPCamadaEscondida;

    private double[] mCamadaSaida = new double[NUMERO_NEURONIOS_CAMADA_SAIDA];

    // Inicializalicao da Rede Neural
    public MLP(double[] entrada, int[] valoresEsperados) {
        /** Default constructor with Number of Epocas and Learning Rate */
        this(entrada, valoresEsperados, 10000, 0.5);
    }

    public MLP(double[] entrada, int[] valoresEsperados, int numberEpocas, double taxaAprendizado) {
        mFin = new FuncaoAtivacao(FuncaoAtivacao.Type.Sigmoide);

        // Inicializa Camada de Entrada
        mCamadaEntrada = entrada;
        mValoresEsperados = valoresEsperados;
        mPCamadaEntrada = randomPesos(mCamadaEntrada);

        // Inicializa Camada Escondida
        mCamadaEscondida = calcularSomatorio(mCamadaEntrada, mPCamadaEntrada, false);
        mPCamadaEscondida = randomPesos(mCamadaEscondida);

        mNumeroEpocas = numberEpocas;
        mTaxaAprendizado = taxaAprendizado;
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
            for(int j = 1; j < erro.length; j++){
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
            erroFinal[k] = aux[k] * mFin.derivate(camadaEscondida[k]);
        }

        return erroFinal;
    }

    /**
     * Atualização dos Pesos
     */
    private static double[][] atualizaPesos(double[][] pcamadaX, double[][] deltapeso, double[] dpbias){
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

    /**
     * Métodos responsáveis por executar a Rede Neural
     */
    public double[] executarEpocas() {
        int j = 0;
        while (mTaxaAprendizado < 1 || j < mNumeroEpocas) {

            executarEpoca();

            mTaxaAprendizado += 0.001;
            j++;
        }
        return mCamadaSaida;
    }

    private void executarEpoca() {
        stepFeedForward();
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
        double[][] deltaPesosCamadaEscondida = deltaPesos(erroSaida, mCamadaEscondida, mCamadaSaida.length);
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